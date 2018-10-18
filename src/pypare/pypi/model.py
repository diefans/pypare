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

import asyncio
import collections
import collections.abc
import functools
import hashlib
from pathlib import Path
import time
import typing

import aiofiles
import aiohttp.web
import aiotask_context
import attr
import inotipy
from packaging import version as pkg_version
import structlog

from .. import utils

log = structlog.get_logger()

DEFAULT_CACHE_TIMEOUT = 60 * 60 * 24
"""Cache timeout defaults to one day."""

CHANNELS_NAME = 'channels'
PROJECTS_NAME = 'projects'
RELEASES_NAME = 'releases'
LATEST_NAME = 'latest'
METADATA_FILENAME = 'metadata.json'
UPSTREAM_METADATA_FILENAME = 'upstream-metadata.json'
FILES_NAME = 'files'
PREPARING_SUFFIX = '.preparing'


def ensure_cls(cls, *containers, key_converter=None):
    """If the attribute is an instance of cls, pass, else try constructing."""
    def converter(val):
        val = val if isinstance(val, cls) else cls(**val)
        return val

    def converter_list(converter, val):
        return [converter(item) for item in val]

    def converter_dict(converter, val):
        return {key_converter(key) if key_converter else key: converter(value)
                for key, value in val.items()}

    if containers:
        for container in reversed(containers):
            if container is list:
                converter = functools.partial(converter_list, converter)

            if container is dict:
                converter = functools.partial(converter_dict, converter)
    return converter


def ensure_semver(val):
    if isinstance(val, pkg_version._BaseVersion):
        return val
    val = pkg_version.parse(val)
    return val


class str_keyed_dict(dict):

    """A str keyed dict."""

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            it, = args
            super().__init__((
                (str(key), value)
                for key, value in (it
                                   if isinstance(it, collections.abc.Iterable)
                                   else it.iter())
            ), **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(str(key), value)


def ensure_parents(path):
    """Create the parent path if not existing."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True)


def symlink_relative_to(from_path, to_path):
    """Create a relative symlink.

    The common base path is search and its distance to from_path is
    substituted with `..` parts and prefixed before to_path.
    """
    from_parts = from_path.parts
    to_parts = to_path.parts
    i = 0
    common_path = None
    while i <= len(from_parts) and i <= len(to_parts):
        if to_parts[i] == from_parts[i]:
            i += 1
            continue
        common_path = Path(*from_parts[:i])
        break

    if common_path:
        prefix_path = Path(
            *['..']
            # parents containes the dot dir as last parent
            # so we start from parent
            * len(from_path.relative_to(common_path).parent.parents)
        )
        to_path = prefix_path / to_path.relative_to(common_path)

    if from_path.exists():
        from_path.unlink()
    from_path.symlink_to(to_path)


class CacheException(Exception):
    pass


class MetadataNotFound(CacheException):
    pass


class MetadataRetrievingError(CacheException):
    pass


@attr.s(kw_only=True, auto_attribs=True)        # noqa: R0903
class Info:
    author: str = ''
    author_email: str = ''
    bugtrack_url: str = ''
    classifiers: typing.List[str] = []
    description: str = ''
    description_content_type: str = ''
    docs_url: typing.List[str] = []
    download_url: str = ''
    downloads: typing.Dict[str, int] = {}
    home_page: str = ''
    keywords: str = ''
    license: str = ''
    maintainer: str = ''
    maintainer_email: str = ''
    name: str
    package_url: str = ''
    platform: str = ''
    project_url: str = ''
    project_urls: typing.List[str] = []
    release_url: str = ''
    requires_dist: typing.List[str] = []
    requires_python: str = ''
    summary: str = ''
    version: pkg_version._BaseVersion = attr.ib(converter=ensure_semver)

    @classmethod
    def from_dict(cls, dct):
        info = cls(**dct)
        return info


@attr.s(kw_only=True, auto_attribs=True)
class Release:
    comment_text: str
    digests: typing.Dict[str, str]
    downloads: int
    filename: str
    has_sig: bool
    md5_digest: str
    packagetype: str
    python_version: str
    requires_python: str
    size: int
    upload_time: str
    url: str

    @classmethod
    def from_dict(cls, dct):
        release = cls(**dct)
        return release

    @utils.reify
    def path_hashs(self):
        """A unique path to the release.

        We assume that the name of the file is unique over project and version
        and packagetype.
        """
        unique_name = ''.join((
            self.filename,
        )).encode('utf-8')
        path_hashs = [
            hashlib.blake2s(unique_name,                       # noqa: E1101
                            digest_size=digest_size,
                            person=b'ChxRel').hexdigest()
            for digest_size in (1, 2, hashlib.blake2s.MAX_DIGEST_SIZE)  # noqa: E1101
        ]
        return path_hashs

    def channel_path(self, channel):
        path = channel.releases_path(*self.path_hashs)
        return path


@attr.s(kw_only=True, auto_attribs=True)
class Metadata:      # noqa: R0903
    info: Info = attr.ib(converter=ensure_cls(Info))
    releases: typing.Dict[str, typing.List[Release]] = attr.ib(
        converter=ensure_cls(
            Release, dict, list, key_converter=ensure_semver),
        factory=dict)
    urls: typing.List[Release] = attr.ib(
        converter=ensure_cls(Release, list),
        factory=list)
    last_serial: int = 0

    @classmethod
    def from_dict(cls, dct):
        project = cls(**dct)
        return project

    def __json__(self):
        dct = attr.asdict(self, dict_factory=str_keyed_dict)
        return dct


class ACL:      # noqa: R0903
    allow = 'access:allow'
    deny = 'accces:deny'
    unauthorized = 'auth:unauthorized'
    authorized = 'auth:authorized'
    all = 'auth:all'
    admin = 'perm:admin'
    group = 'perm:group'
    read = 'perm:read'
    write = 'perm:write'


@attr.s(kw_only=True, auto_attribs=True)        # noqa: R0903
class Cache:        # noqa: R0903
    root: Path = attr.ib(converter=Path)

    def path(self, *path):
        """Return a path relativ to the cache root."""
        return self.root / Path(*path)

    async def channel(self, name, **kwargs):
        channel = await Channel.from_cache(self, name=name, **kwargs)
        return channel


@attr.s(auto_attribs=True)        # noqa: R0903
class ACE:
    permits: bool
    principal: str
    permission: str


@attr.s(kw_only=True, auto_attribs=True)        # noqa: R0903
class Channel(Cache):
    name: str = 'pypi'
    acl: typing.List[ACE] = [ACE(ACL.allow, ACL.unauthorized, ACL.read)]
    timeout: float = DEFAULT_CACHE_TIMEOUT
    upstream_enabled: bool = True
    upstream_api_base: str = 'https://pypi.org/pypi'
    releases_route: aiohttp.web.AbstractRoute
    """The route which handles releases requests."""

    @classmethod
    async def from_cache(cls, cache, *, name, **kwargs):
        root = cache.path(CHANNELS_NAME, name)
        channel = await cls.from_path(root, name=name, **kwargs)
        return channel

    @classmethod
    def from_cache_only(cls, cache, *, name, **kwargs):
        root = cache.path(CHANNELS_NAME, name)
        channel = cls(root=root, **kwargs)
        return channel

    @classmethod
    async def from_path(cls, root, **kwargs):
        async with aiofiles.open(root / METADATA_FILENAME) as f:
            data = utils.json_loads(await f.read())
            data.update(kwargs)
            channel = cls(root=root, **data)
            return channel

    async def store(self):
        data = attr.asdict(self, dict_factory=str_keyed_dict)
        # sanitize
        del data['releases_route']
        del data['root']

        metadata_path = self.root / METADATA_FILENAME
        ensure_parents(metadata_path)
        async with aiofiles.open(metadata_path, 'x') as f:
            await f.write(utils.json_dumps(data))

    async def release(self, path):
        """The release for this path."""
        release_path = self.releases_path(path) / METADATA_FILENAME
        async with aiofiles.open(release_path) as f:
            data = utils.json_loads(await f.read())
            release = Release.from_dict(data)
            return release

    async def upstream_release(self, path):
        """The upstream release for this path."""
        release_path = self.releases_path(path) / UPSTREAM_METADATA_FILENAME
        async with aiofiles.open(release_path) as f:
            data = utils.json_loads(await f.read())
            release = Release.from_dict(data)
            return release

    def projects_path(self, *path):
        projects_path = self.path(PROJECTS_NAME, *path)
        return projects_path

    def releases_path(self, *path):
        releases_path = self.path(RELEASES_NAME, *path)
        return releases_path

    def project(self, project_name):
        """Retrun the project.

        If this channel is an upstream channel, we create an
        :py:obj:`UpstreamProject`
        """
        if self.upstream_enabled:
            project = UpstreamProject(channel=self, name=project_name,
                                      api_base=self.upstream_api_base)
        else:
            project = Project(channel=self, name=project_name)
        return project

    async def store_project(self, data):
        """
        Create or update a projects metadata structures.

        :param data: the raw parsed metadata, as the json API from pypi
        provides it.

        channels/<channel>
            |
            +- projects/<project[0]>/<project>/<version>
            |   |
            |   +- metadata.json
            |   |
            |   +- files/<packagetype>.json <-+
            |                                 |
            +- releases                       |
                |                             |
                +- <path>/metadata.json ------+
        """
        metadata = Metadata.from_dict(data)
        project_name = metadata.info.name
        project = self.project(project_name)

        await project.store_metadata(metadata)


@attr.s(kw_only=True, auto_attribs=True)
class Project:

    """A Project embodies a pypi project."""

    log = structlog.get_logger(':'.join((__module__, __qualname__)))    # noqa: E0602

    channel: Channel
    name: str

    @utils.reify
    def path(self):
        """The project root path."""
        path = self.channel.projects_path(self.name[0], self.name)
        return path

    @utils.reify
    def path_latest(self):
        """The directory of the latest version."""
        return self.path / LATEST_NAME

    @property
    def latest_version(self):
        """The version number of the latest version."""
        if not self.path_latest.is_symlink():
            return None
        name = self.path_latest.resolve().name
        version = pkg_version.parse(name)
        return version

    def version_path(self, version):
        """The path of a version."""
        # we need to cast version nto str
        path = self.path / str(version)
        return path

    def info_path(self, version):
        """The path to the metadata of a version."""
        path = self.version_path(version) / METADATA_FILENAME
        return path

    async def load_metadata(self, version=None):
        """Return the metadata for that project and version."""
        if version is None:
            version = self.latest_version

        info_path = self.info_path(version)
        async with aiofiles.open(info_path) as f:
            data = await f.read()
            info = Info.from_dict(utils.json_loads(data))
            self.log.debug('Loaded project info', path=info_path)
        # contains all version releases
        releases = collections.defaultdict(list)
        # contans only releases of the selected version
        urls = []
        # find all <version>/files/*.json files
        for version_path in self.path.iterdir():
            name = version_path.name
            if name in ('latest',):
                continue
            found_version = pkg_version.parse(name)

            for path in version_path.glob('files/*.json'):
                async with aiofiles.open(path) as f:
                    data = await f.read()
                    release = Release.from_dict(utils.json_loads(data))
                    self.log.debug('Loaded release metadata', path=path,
                                   version=found_version)
                    releases[found_version].append(release)
                    if found_version == version:
                        urls.append(release)

        metadata = Metadata(
            info=info,
            releases=releases,
            urls=urls
        )
        return metadata

    async def store_metadata(self, metadata):
        self.log.debug('Update metadata', project=self)
        await self.store_info(metadata.info)
        await self.store_releases(metadata.releases)

    async def store_info(self, info):
        # lookup the latest info
        version_path = self.version_path(info.version)
        info_path = version_path / METADATA_FILENAME
        ensure_parents(info_path)
        async with aiofiles.open(info_path, 'w') as f:
            data = utils.json_dumps(attr.asdict(info))
            await f.write(data)
            self.log.debug('Stored project info', name=self.name,
                           path=info_path, version=info.version)

        if self.latest_version is None or info.version > self.latest_version:
            symlink_relative_to(self.path_latest, version_path)

    async def store_releases(self, releases):
        for version, packages in releases.items():
            for release in packages:
                await self.store_release(version, release)

    async def store_release(self, version, release):
        """Update a package release."""
        version_path = self.version_path(version)
        release_metadata_path = (
            version_path / FILES_NAME / release.filename
        ).with_suffix('.json')
        ensure_parents(release_metadata_path)

        # write metadata
        self.log.debug('Storing release metadata', version=version,
                       filename=release.filename,
                       path=release_metadata_path)
        async with aiofiles.open(release_metadata_path, 'w') as f:
            data = utils.json_dumps(attr.asdict(release,
                                                dict_factory=str_keyed_dict))
            await f.write(data)
            self.log.debug('Stored release metadata', version=version,
                           filename=release.filename,
                           path=release_metadata_path)

        # link metadata to release path
        channel_release_path = release.channel_path(self.channel)
        release_path_metadata_link = channel_release_path / METADATA_FILENAME
        ensure_parents(release_path_metadata_link)
        symlink_relative_to(release_path_metadata_link, release_metadata_path)


@attr.s(kw_only=True, auto_attribs=True)
class UpstreamProject(Project):

    log = structlog.get_logger(':'.join((__module__, __qualname__)))    # noqa: E0602

    api_base: str = 'https://pypi.org/pypi'

    @property
    def mtime(self):
        try:
            max_mtime = max(path.stat().st_mtime
                            for path in self.path.glob('**/*')
                            if path.is_file())
            return max_mtime
        except ValueError:
            return None

    def url(self, version=None):
        """The upstream url for a or the latest version."""
        def _gen():
            yield self.api_base
            yield self.name
            if version:
                yield str(version)
            yield 'json'
        url = '/'.join(_gen())
        return url

    def needs_update(self, version=None):
        """Either there is no latest version or the timeout is over."""
        mtime = self.mtime
        timeout = ((time.time() - mtime) >= self.channel.timeout
                   if mtime else True)

        return timeout or not self.info_path(version).exists()

    async def load_metadata(self, version=None):
        if not self.needs_update:
            metadata = await super().load_metadata(version)
            return metadata

        # update and cache
        url = self.url(version)
        client_session = aiotask_context.get('client_session')
        self.log.info('Loading upstream metadata', url=url)
        async with client_session.get(url) as r:
            if r.status == 200:
                data = await r.json()
                metadata = Metadata.from_dict(data)
                await self.store_metadata(metadata)
                self.log.info('Loading metadata from upstream', response=r)
                metadata = await super().load_metadata(version)
                return metadata

            if r.status == 404:
                log.error('Metadata not found', project=self, response=r)
                raise MetadataNotFound(self, r)

            self.log.error('Error while retrieving metadata', project=self,
                           response=r)
            raise MetadataRetrievingError(self, r)

    async def store_release(self, version, release):
        """Update a package release.

        The release url is transformed to point to ourself, but the original
        upstream metadata are stored too for later processing.

        """
        # create the our own url
        channel_release_path = release.channel_path(self.channel)
        # store upstream metadata
        upstream_release_metadata_path \
            = channel_release_path / UPSTREAM_METADATA_FILENAME
        ensure_parents(upstream_release_metadata_path)

        self.log.debug('Storing upstream release metadata', version=version,
                       path=upstream_release_metadata_path)
        async with aiofiles.open(upstream_release_metadata_path, 'w') as f:
            data = utils.json_dumps(attr.asdict(release,
                                                dict_factory=str_keyed_dict))
            await f.write(data)
            self.log.debug('Stored upstream release metadata', version=version,
                           path=upstream_release_metadata_path)

        # inject our own url
        release.url = self.release_url(release)
        await super().store_release(version, release)

    def release_url(self, release):
        """Create the release url for the actual configuration."""
        path = Path('/') / Path(*release.path_hashs)
        url = self.channel.releases_route.url_for(
            channel_name=self.channel.name,
            path=str(path.relative_to('/')),
            filename=release.filename
        )
        return url


@attr.s(kw_only=True, auto_attribs=True)
class CachingStreamer:

    log = structlog.get_logger(':'.join((__module__, __qualname__)))    # noqa: E0602

    url: str
    path: Path
    upstream_enabled: bool

    @classmethod
    async def from_channel_release_url(cls, channel, path, filename):
        release = await channel.release(path)
        upstream_release = await channel.upstream_release(path)
        streamer = cls(
            url=upstream_release.url,
            path=release.channel_path(channel) / filename,
            upstream_enabled=channel.upstream_enabled
        )
        return streamer

    @utils.reify
    def path_preparing(self):
        path = self.path.with_name(self.path.name + PREPARING_SUFFIX)
        return path

    async def stream(self, force=False):
        """Stream a released version file.

        If there is a cache, we assume this release version is final and we
        stream from it. Otherwise we download the data, store and stream it.

        :param force: force recaching
        """
        if not self.path.is_file() or force:
            if self.path_preparing.is_file():
                async def _gen():
                    # serve from intermediate file
                    async for data in self._stream_from_intermediate():
                        yield data
            elif self.upstream_enabled:
                # testing for 404
                client_session = aiotask_context.get('client_session')
                async with client_session.head(self.url) as r:
                    if r.status != 200:
                        if r.status == 404:
                            raise aiohttp.web.HTTPNotFound()
                        raise aiohttp.web.HTTPInternalServerError()

                async def _gen():
                    async for data in self._stream_and_cache():
                        yield data
            else:
                raise aiohttp.web.HTTPNotFound()
        else:
            async def _gen():
                self.log.info('Serving cache', path=self.path)
                async with aiofiles.open(self.path, 'rb') as f:
                    async for data in ChunkedFileIterator(f, chunk_size=2**14):
                        self.log.debug('Stream data', size=len(data))
                        yield data
        return _gen()

    async def _stream_and_cache(self):
        """Stream data from upstream and cache them.

        The download and caching is done in the background, to prevent
        disconnecting clients from stopping it.
        """
        client_session = aiotask_context.get('client_session')
        self.log.info('Caching upstream', url=self.url, path=self.path)

        queue = asyncio.Queue()
        fut_finished = asyncio.Future()
        cur_task = asyncio.current_task()

        async def _stream_queue():
            while queue.qsize() or not fut_finished.done():
                data = await queue.get()
                try:
                    yield data
                finally:
                    queue.task_done()

        async def _enqueue_upstream():
            try:
                log.debug('Streaming from upstream into file and queue',
                          file=self.path_preparing, url=self.url)
                async with aiofiles.open(self.path_preparing, 'xb') as f:
                    async with client_session.get(self.url) as r:
                        async for data, _ in r.content.iter_chunks():
                            await f.write(data)
                            await queue.put(data)
                        fut_finished.set_result(True)
                self.path_preparing.rename(self.path)
                self.log.info('Finished download', path=self.path)
            except (asyncio.CancelledError, IOError, Exception) as ex:      # noqa: W0703
                cur_task.cancel()
                # cleanup broken download
                self.log.error('Cleaning broken download',
                               path=self.path_preparing, error=ex)
                try:
                    self.path_preparing.unlink()
                except FileNotFoundError:
                    pass

        # TODO use aiojobs ??? to cancel this future graceully
        # GeneratorExit
        asyncio.ensure_future(_enqueue_upstream())
        async for data in _stream_queue():
            yield data

    async def _stream_from_intermediate(self):
        self.log.info('Stream from intermediate',
                      path=self.path_preparing)

        watcher = inotipy.Watcher.create()
        watcher.watch(str(self.path_preparing),
                      inotipy.IN.MOVE_SELF
                      | inotipy.IN.DELETE_SELF
                      | inotipy.IN.CLOSE_WRITE
                      | inotipy.IN.MODIFY
                      )
        fut_finished = asyncio.Future()
        ev_write = asyncio.Event()

        async def _wait_for_event():
            while True:
                event = await watcher.get()
                self.log.debug('File event',
                               file_event=event, watch=event.watch)
                if event.mask & inotipy.IN.MODIFY:
                    ev_write.set()
                if event.mask & inotipy.IN.DELETE_SELF:
                    fut_finished.set_exception(FileNotFoundError(event))
                    break

                if event.mask & (
                        inotipy.IN.MOVE_SELF | inotipy.IN.CLOSE_WRITE):
                    fut_finished.set_result(event)
                    break

        async with aiofiles.open(self.path_preparing, 'rb') as f:
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

    async def response(self):
        headers = {
            'Content-disposition':
            f'attachment; filename={self.path.name}'
        }
        stream = await self.stream()
        return aiohttp.web.Response(
            body=stream,
            headers=headers
        )


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
