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
import abc
import asyncio
import hashlib
import pathlib
import re
import time
import urllib

import aiofiles
import aiohttp
import aiohttp.web
import aiohttp_jinja2
import aiotask_context
import attr
import jinja2
import pkg_resources
import structlog

from . import utils, inotify

log = structlog.get_logger()
routes = aiohttp.web.RouteTableDef()
re_semver = re.compile(
    r'\bv?(?:0|[1-9]\d*)'
    r'\.(?:0|[1-9]\d*)'
    r'\.(?:0|[1-9]\d*)'
    r'(?:-[\da-z-]+(?:\.[\da-z-]+)*)'
    r'?(?:\+[\da-z-]+(?:\.[\da-z-]+)*)?\b')

default_cache_timeout = 60 * 60 * 24
"""Cache timeout defaults to one day."""


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


@routes.get(r'/release/{path:[a-f0-9/]+}/{file}', name='release')
async def serve_release(request):
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


@routes.get(r'/{project_name}')
@routes.get(r'/{project_name}/')
@routes.get(r'/{project_name}/{json:json}')
@routes.get(r'/{project_name}/{version:' + re_semver.pattern + r'}')
@routes.get(r'/{project_name}/{version:' + re_semver.pattern + r'}/')
@routes.get(r'/{project_name}/{version:' + re_semver.pattern + r'}/{json:json}')    # noqa
async def get_project(request):
    pypi_cache = request.config_dict['pypi_cache']
    project_name = request.match_info['project_name']
    version = request.match_info.get('version', None)
    project = CachedProject(project_name, version, cache=pypi_cache)

    try:
        metadata = await project.get_metadata()
    except MetadataNotFound:
        raise aiohttp.web.HTTPNotFound()
    except MetadataRetrievingError:
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


class CacheException(Exception):
    pass


class MetadataNotFound(CacheException):
    pass


class MetadataRetrievingError(CacheException):
    pass


class Cache:        # noqa: R0903
    def __init__(self, cache_root, cache_timeout=None):
        self.cache_root = cache_root
        self.cache_timeout = cache_timeout or default_cache_timeout

    def principal_project_path(self, project_name, principal=None):
        path = self.cache_root.joinpath(
            'principals',
            'default' if principal is None else principal,
            project_name[0],
            project_name
        )
        return path

    def project(self, project_name):
        project = CachedProject(project_name, cache=self)
        return project


class Project:      # noqa: R0903
    def __init__(self, project_name, version=None):
        self.project_name = project_name
        self.version = version

    @abc.abstractmethod
    async def get_metadata(self):
        """Return the metadata for that project and version."""


class PypiProject(Project):
    api_base = 'https://pypi.org/pypi/{project_version}/json'.format

    @utils.reify
    def url(self):
        project_version = (self.project_name
                           if self.version is None
                           else '/'.join((self.project_name, self.version)))
        return self.api_base(project_version=project_version)

    async def get_metadata(self):
        client_session = aiotask_context.get('client_session')
        async with client_session.get(self.url) as r:
            if r.status == 200:
                log.info('Loading metadata from upstream', url=self.url)
                metadata = await r.json()
                return metadata

            if r.status == 404:
                log.error('Metadata not found', project=self)
                raise MetadataNotFound(self, r)

            log.error('Error while retrieving metadata', project=self)
            raise MetadataRetrievingError(self, r)


class CachedProject(PypiProject):

    """A Project is a cached pypi project."""

    def __init__(self, project_name, version=None, *, cache):
        super().__init__(project_name, version)
        self.cache = cache

    @utils.reify
    def cache_path(self):
        path = self.cache.principal_project_path(self.project_name)
        if not path.is_dir():
            path.mkdir(parents=True)
        return path

    @utils.reify
    def cached_metadata_upstream_path(self):
        """The cached version of upstream metadata."""
        name = ('pypi-metadata.json'
                if self.version is None
                else f'pypi-metadata-{self.version}.json')
        path = self.cache_path.joinpath(name)
        return path

    @utils.reify
    def cached_metadata_path(self):
        """Our transformed metadata.

        We change urls of releases here.
        """
        name = ('metadata.json'
                if self.version is None
                else f'metadata-{self.version}.json')
        path = self.cache_path.joinpath(name)
        return path

    async def get_metadata(self):
        """Either lookup in cache or query pypi and store in cache."""
        # try to find actual versions if not modified since cache_timeout
        path = self.cached_metadata_path
        if (path.is_file()
                and (time.time() - path.stat().st_mtime)
                < self.cache.cache_timeout):
            # serve this file
            async with aiofiles.open(path) as f:
                log.info('Loading metadata from cache', path=path)
                metadata = utils.json_loads(await f.read())
        else:
            # create cache
            upstream_metadata = await super().get_metadata()
            async with aiofiles.open(
                    self.cached_metadata_upstream_path, "w") as f:
                await f.write(utils.json_dumps(upstream_metadata))
            # transform upstream
            metadata = await self.transform_metadata(upstream_metadata)
            async with aiofiles.open(self.cached_metadata_path, "w") as f:
                await f.write(utils.json_dumps(metadata))
        return metadata

    async def transform_metadata(self, upstream_metadata):
        """Inject our own release URL into upstream metadata."""

        async def _release_urls(urls):
            urls = [await self.transform_upstream_url(url,
                                                      prepare_cache=True)
                    for url in urls]
            return urls

        # TODO XXX FIXME
        # noqa file a bug: SyntaxError: asynchronous comprehension outside of an asynchronous function
        releases = {
            version: await _release_urls(urls)
            for version, urls in upstream_metadata['releases'].items()
        }
        urls = [
            await self.transform_upstream_url(url)
            for url in upstream_metadata['urls']
        ]

        metadata = {
            'info': upstream_metadata['info'],
            'last_serial': upstream_metadata['last_serial'],
            'releases': releases,
            # files for this version
            'urls': urls
        }
        return metadata

    async def transform_upstream_url(self, url, prepare_cache=False):
        """Transform the upstream url and prepare cache.

        Since pypi sends multiple equal urls in metadata, we do not want to
        always prepare the cache.

        """
        cached_release = CachedRelease(cache=self.cache, url=url)
        if prepare_cache:
            # TODO prepare cache by examining cache_path for existing metadata
            await cached_release.prepare()

        transformed_url = await cached_release.transform_url()

        return transformed_url


@attr.s
class CachedRelease:
    cache = attr.ib()
    url = attr.ib(default=None)

    @utils.reify
    def upstream_url(self):
        upstream_url = self.url['url']
        return upstream_url

    @utils.reify
    def cache_path(self):
        path = self.cache.cache_root.joinpath('releases')
        return path

    @utils.reify
    def path(self):
        encoded_upstream_url = self.upstream_url.encode('utf-8')
        upstream_url_hashs = [
            hashlib.blake2s(encoded_upstream_url,                       # noqa: E1101
                            digest_size=digest_size,
                            person=b'cacheReL').hexdigest()
            for digest_size in (1, 2, hashlib.blake2s.MAX_DIGEST_SIZE)  # noqa: E1101
        ]
        path = pathlib.Path().joinpath(*upstream_url_hashs)
        return path

    @utils.reify
    def file_name(self):
        parsed_upstream_url = urllib.parse.urlparse(self.upstream_url)
        file_name = pathlib.Path(parsed_upstream_url.path).name
        return file_name

    @utils.reify
    def release_file_path(self):
        path = self.cache_path.joinpath(self.path, self.file_name)
        return path

    @utils.reify
    def release_file_path_preparing(self):
        path = self.release_file_path.with_name(
            self.release_file_path.name + '.preparing')
        return path

    async def transform_url(self):
        url = self.url.copy()
        request = aiotask_context.get('request')

        url['url'] = request.app.router['release'].url_for(
            path=str(self.path),
            file=self.file_name
        )
        return url

    def metadata_path(self, path):
        metadata_path = self.cache_path.joinpath(path, 'metadata.json')
        return metadata_path

    async def prepare(self):
        # copy url metadata to cache path
        metadata_path = self.metadata_path(self.path)
        if not metadata_path.parent.is_dir():
            metadata_path.parent.mkdir(parents=True)

        async with aiofiles.open(metadata_path, 'w') as f:
            data = utils.json_dumps(self.url)
            await f.write(data)
            log.info('Cached url metadata', path=metadata_path)

    @classmethod
    async def from_cached_metadata(cls, cache, path):
        cached_release = cls(cache)
        metadata_path = cached_release.metadata_path(path)
        async with aiofiles.open(metadata_path) as f:
            data = await f.read()
            cached_release.url = utils.json_loads(data)
        return cached_release

    @property
    def streamer(self):
        streamer = CachingStreamer(self.upstream_url, self.release_file_path)
        return streamer


class CachingStreamer:

    log = structlog.get_logger(':'.join((__module__, __qualname__)))    # noqa: E0602

    def __init__(self, url, file_path):
        self.url = url
        self.file_path = file_path

    @utils.reify
    def file_path_preparing(self):
        path = self.file_path.with_name(
            self.file_path.name + '.preparing')
        return path

    async def stream(self, force=False):
        """Stream a released version file.

        If there is a cache, we assume this release version is final and we
        stream from it. Otherwise we download the data, store and stream it.

        :param force: force recaching
        """
        if not self.file_path.is_file() or force:
            if self.file_path_preparing.is_file():
                # serve from intermediate file
                async for data in self._stream_from_intermediate():
                    self.log.debug('Stream data', size=len(data))
                    yield data
            else:
                async for data in self._stream_and_cache():
                    self.log.debug('Stream data', size=len(data))
                    yield data
        else:
            self.log.info('Serving cache', path=self.file_path)
            async with aiofiles.open(self.file_path, 'rb') as f:
                async for data in ChunkedFileIterator(f, chunk_size=2**14):
                    self.log.debug('Stream data', size=len(data))
                    yield data

    async def _stream_and_cache(self):
        """Stream data from upstream and cache them.

        The download and caching is done in the background, to prevent
        disconnecting clients from stopping it.
        """
        client_session = aiotask_context.get('client_session')
        self.log.info('Caching upstream', url=self.url, path=self.file_path)

        queue = asyncio.Queue()
        fut_finished = asyncio.Future()

        async def _stream_queue():
            while queue.qsize() or not fut_finished.done():
                data = await queue.get()
                try:
                    yield data
                finally:
                    queue.task_done()

        async def _enqueue_upstream():
            try:
                async with aiofiles.open(self.file_path_preparing, 'xb') as f:
                    async with client_session.get(self.url) as r:
                        async for data, _ in r.content.iter_chunks():
                            await f.write(data)
                            await queue.put(data)
                        fut_finished.set_result(True)
                self.file_path_preparing.rename(self.file_path)
                self.log.info('Finished download', path=self.file_path)
            except (asyncio.CancelledError, IOError, Exception) as ex:      # noqa: W0703
                fut_finished.set_exception(ex)
                # cleanup broken download
                self.log.error('Cleaning broken download',
                               path=self.file_path_preparing, error=ex)
                try:
                    self.file_path_preparing.unlink()
                except FileNotFoundError:
                    pass

        # TODO use aiojobs ??? to cancel this future graceully
        # GeneratorExit
        asyncio.ensure_future(_enqueue_upstream())
        async for data in _stream_queue():
            yield data

    async def _stream_from_intermediate(self):
        self.log.info('Stream from intermediate',
                      path=self.file_path_preparing)

        watcher = inotify.Watcher.create()
        watcher.watch(str(self.file_path_preparing),
                      inotify.IN.MOVE_SELF
                      | inotify.IN.DELETE_SELF
                      | inotify.IN.CLOSE_WRITE
                      | inotify.IN.MODIFY
                      )
        fut_finished = asyncio.Future()
        ev_write = asyncio.Event()

        async def _wait_for_event():
            while True:
                event = await watcher.get()
                self.log.debug('File event',
                               file_event=event, watch=event.watch)
                if event.mask & inotify.IN.MODIFY:
                    ev_write.set()
                if event.mask & inotify.IN.DELETE_SELF:
                    fut_finished.set_exception(FileNotFoundError(event))
                    break

                if event.mask & (
                        inotify.IN.MOVE_SELF | inotify.IN.CLOSE_WRITE):
                    fut_finished.set_result(event)
                    break

        async with aiofiles.open(self.file_path_preparing, 'rb') as f:
            while True:
                data = await f.read()
                if data:
                    yield data
                elif fut_finished.done():
                    self.log.info('Intermediate finished',
                                  result=await fut_finished)
                    break
                else:
                    # wait for next write event
                    await ev_write.wait()
                    ev_write.clear()


class ChunkedFileIterator:

    """Iterate and yield binary data chunks from a file.

    :param file: the file
    :param chunk_size: the size of the chunks to yield

    .. code-block:: python

        async for data in ChunkedFileIterator(f):
            yield data

    """
    def __init__(self, file, chunk_size=2**14):
        self.file = file
        self.chunk_size = chunk_size

    def __aiter__(self):
        """We are our own iterator."""
        return self

    async def __anext__(self):
        """Simulate normal file iteration."""
        chunk = await self.file.read(self.chunk_size)
        if chunk:
            return chunk
        raise StopAsyncIteration


def jinja2_filter_basename(path):
    return pathlib.Path(str(path)).name


def jinja2_filter_dirname(path):
    return pathlib.Path(str(path)).parent


async def plug_me_in(app):
    await app.plugin('pypare.plugins:plugin_task_context')
    pypi_cache = Cache(
        cache_root=app.config['cache_root'],
        cache_timeout=app.config['cache_timeout'],
    )
    log.info('Using cache', cache_root=pypi_cache.cache_root,
             cache_timeout=pypi_cache.cache_timeout)

    pypi_app = aiohttp.web.Application()
    pypi_app['pypi_cache'] = pypi_cache
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
