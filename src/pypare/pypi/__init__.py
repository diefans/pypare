# Copyright 2018 Oliver Berger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Very simple caching pypi proxy."""
from pathlib import Path

import aiohttp
import aiohttp.web
import aiohttp_jinja2
import jinja2
from packaging.version import VERSION_PATTERN
import structlog

from . import model
from .. import utils

RELEASES_ROUTE = 'releases'

log = structlog.get_logger()
routes = aiohttp.web.RouteTableDef()


def strip_verbose_pattern(pattern):
    def _gen():
        for line in pattern.split('\n'):
            stripped, *_ = line.rsplit('#', 1)
            yield stripped.strip()

    return r''.join(_gen())


VERSION_PATTERN = strip_verbose_pattern(VERSION_PATTERN)


@routes.get('/')
async def index(request):
    config = request.config_dict['config']
    context = {
        'projects': [],
        'principal': 'default',
    }
    response = aiohttp_jinja2.render_template(
        'simple.j2', request, context)
    return response


@routes.get(r'/{channel_name}/+releases/{path:[a-f0-9/]+}/{filename}',
            name=RELEASES_ROUTE)
async def serve_release(request):
    channel_name = request.match_info['channel_name']
    path = request.match_info['path']
    filename = request.match_info['filename']
    cache = request.config_dict['cache']

    channel = await cache.channel(
        name=channel_name,
        releases_route=request.app.router[RELEASES_ROUTE]
    )
    streamer = await model.CachingStreamer.from_channel_release_url(
        channel, path, filename
    )
    return await streamer.response()


@routes.get(r'/{channel_name}/{project_name}')
@routes.get(r'/{channel_name}/{project_name}/')
@routes.get(r'/{channel_name}/{project_name}/{json:json}')
@routes.get(r'/{channel_name}/{project_name}/{version:' + VERSION_PATTERN + r'}')               # noqa
@routes.get(r'/{channel_name}/{project_name}/{version:' + VERSION_PATTERN + r'}/')              # noqa
@routes.get(r'/{channel_name}/{project_name}/{version:' + VERSION_PATTERN + r'}/{json:json}')   # noqa
async def get_project(request):
    channel_name = request.match_info['channel_name']
    project_name = request.match_info['project_name']
    version = request.match_info.get('version', None)
    version = model.pkg_version.parse(version) if version else None
    cache = request.config_dict['cache']

    channel = await cache.channel(
        name=channel_name,
        releases_route=request.app.router[RELEASES_ROUTE]
    )
    project = channel.project(project_name)

    try:
        metadata = await project.load_metadata(version)
    except model.MetadataNotFound:
        raise aiohttp.web.HTTPNotFound()
    except model.MetadataRetrievingError:
        raise aiohttp.web.HTTPInternalServerError()

    # if we have a json request, like the new pypi API
    if request.match_info.get('json', False):
        return utils.json_response(metadata)

    context = {
        'project_name': project_name,
        'metadata': metadata,
    }
    response = aiohttp_jinja2.render_template(
        'simple_package.j2', request, context)
    return response


def jinja2_filter_basename(path):
    return Path(str(path)).name


def jinja2_filter_dirname(path):
    return Path(str(path)).parent


async def plug_me_in(app):
    defaults = {
        'upstream_channel_name': 'pypi',
        'upstream_channel_api_base': 'https://pypi.org/pypi',
        'upstream_channel_timeout': 60 * 60 * 24,
    }
    app.config.set_defaults(defaults)

    await app.plugin('pypare.plugins:plugin_task_context')

    # create pypi subapp
    pypi_app = app.config['pypi_app'] = aiohttp.web.Application()
    cache = model.Cache(root=app.config['cache_root'])
    log.info('Using cache', cache_root=cache.root)
    pypi_app['cache'] = cache
    pypi_app.add_routes(routes)
    pypi_app['releases_route'] = pypi_app.router[RELEASES_ROUTE]

    # setup default upstream channel
    # TODO should we really take upstream config from CLI???
    # creates the conflict of overriding an already persistent upstream config
    # better to provide another CLI command for creating channels at all
    upstream_channel_name = app.config['upstream_channel_name']
    try:
        upstream_channel = await cache.channel(
            upstream_channel_name,
            releases_route=pypi_app['releases_route'],
        )
    except FileNotFoundError:
        # create upstream channel
        upstream_channel = model.Channel.from_cache_only(
            cache, name=upstream_channel_name,
            releases_route=pypi_app['releases_route'],
            upstream_api_base=app.config['upstream_channel_api_base'],
            timeout=app.config['upstream_channel_timeout'],
        )
        await upstream_channel.store()
        log.info('Created default upstream channel',
                 name=upstream_channel_name)
    app['upstream_channel'] = upstream_channel

    template_path = str(Path(__spec__.origin).parent / 'templates')
    aiohttp_jinja2.setup(
        pypi_app,
        loader=jinja2.FileSystemLoader(template_path),
        filters={
            'basename': jinja2_filter_basename,
            'dirname': jinja2_filter_dirname,
        },
    )
    app.add_subapp(app.config['base_path'], pypi_app)
    log.info('Added subapp', prefix=app.config['base_path'],
             resources=pypi_app.router._resources)
