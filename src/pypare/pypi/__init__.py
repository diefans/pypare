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
import pathlib
import re

import aiohttp
import aiohttp.web
import aiohttp_jinja2
import jinja2
import pkg_resources
import structlog

from .. import utils
from . import model

log = structlog.get_logger()
routes = aiohttp.web.RouteTableDef()
re_semver = re.compile(
    r'\bv?(?:0|[1-9]\d*)'
    r'\.(?:0|[1-9]\d*)'
    r'\.(?:0|[1-9]\d*)'
    r'(?:-[\da-z-]+(?:\.[\da-z-]+)*)'
    r'?(?:\+[\da-z-]+(?:\.[\da-z-]+)*)?\b')


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


@routes.get(r'/{channel_name}/release/{path:[a-f0-9/]+}/{file}', name='release')
async def serve_release(request):
    channel_name = request.match_info['channel_name']
    channel = model.Channel(name=channel_name)

    path = request.match_info['path']
    release_file = ReleaseFile.from_path(path)

    pypi_cache = request.config_dict['pypi_cache']
    cached_release = await CachedRelease.from_cached_metadata(
        cache=pypi_cache,
        path=pathlib.Path(request.match_info['path'])
    )
    headers = {
        'Content-disposition':
        f'attachment; filename={cached_release.file_name}'
    }
    return aiohttp.web.Response(
        body=cached_release.streamer.stream(),
        headers=headers
    )


@routes.get(r'/{channel_name}/{project_name}')
@routes.get(r'/{channel_name}/{project_name}/')
@routes.get(r'/{channel_name}/{project_name}/{json:json}')
@routes.get(r'/{channel_name}/{project_name}/{version:' + re_semver.pattern + r'}')
@routes.get(r'/{channel_name}/{project_name}/{version:' + re_semver.pattern + r'}/')
@routes.get(r'/{channel_name}/{project_name}/{version:' + re_semver.pattern + r'}/{json:json}')    # noqa
async def get_project(request):
    channel_name = request.match_info['channel_name']
    project_name = request.match_info['project_name']
    version = request.match_info.get('version', None)
    cache = request.config_dict['cache']

    channel = model.Channel(cache=cache, name=channel_name)
    project = model.CachedProject(
        channel=channel,
        name=project_name,
        version=version)

    try:
        metadata = await project.get_metadata()
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
    return pathlib.Path(str(path)).name


def jinja2_filter_dirname(path):
    return pathlib.Path(str(path)).parent


async def plug_me_in(app):
    await app.plugin('pypare.plugins:plugin_task_context')
    cache = model.Cache(
        root=app.config['cache_root'],
        timeout=app.config['cache_timeout'],
    )
    log.info('Using cache', cache_root=cache.root,
             cache_timeout=cache.timeout)

    pypi_app = aiohttp.web.Application()
    pypi_app['cache'] = cache
    pypi_app.add_routes(routes)

    template_path = pkg_resources.resource_filename(__package__, 'templates')
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
