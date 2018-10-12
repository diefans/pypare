import abc
import asyncio
import collections
import functools
import hashlib
import pathlib
import time
import typing
import urllib

import aiofiles
import aiotask_context
import attr
import inotipy
import semantic_version
import structlog

from . import utils

log = structlog.get_logger()

DEFAULT_CACHE_TIMEOUT = 60 * 60 * 24
"""Cache timeout defaults to one day."""

CHANNELS_NAME = 'channels'
PROJECTS_NAME = 'projects'
RELEASES_NAME = 'releases'
LATEST_NAME = 'latest'
METADATA_FILENAME = 'metadata.json'
FILES_NAME = 'files'


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
    if isinstance(val, semantic_version.Version):
        return val
    val = semantic_version.Version(val)
    return val


class str_keyed_dict(dict):

    """A str keyed dict."""

    def __setitem__(self, key, value):
        super().__setitem__(str(key), value)


def ensure_parents(path):
    """Create the parent path if not existing."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True)


def parse_url_path(url):
    """Get the path of the url."""
    parsed_url = urllib.parse.urlparse(url)
    path = pathlib.Path(parsed_url.path)
    return path


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
        common_path = pathlib.Path(*from_parts[:i])
        break

    if common_path:
        prefix_path = pathlib.Path(
            *['..']
            # parents containes the dot dir as last parent
            # so we start from parent
            * len(from_path.relative_to(common_path).parent.parents)
        )
        to_path = prefix_path.joinpath(to_path.relative_to(common_path))
    from_path.symlink_to(to_path)


class CacheException(Exception):
    pass


class MetadataNotFound(CacheException):
    pass


class MetadataRetrievingError(CacheException):
    pass


class ACL:      # noqa: R0903
    allow = 'access:allow'
    deny = 'accces:deny'
    unauthorized = 'auth:unauthorized'
    authorized = 'auth:authorized'
    admin = 'perm:admin'
    group = 'perm:group'
    read = 'perm:read'
    write = 'perm:write'
    all = 'perm:all'


@attr.s(kw_only=True, auto_attribs=True)        # noqa: R0903
class Cache:        # noqa: R0903
    root: pathlib.Path = attr.ib(converter=pathlib.Path)

    def path(self, *path):
        """Return a path relativ to the cache root."""
        return self.root.joinpath(*path)

    def channel(self, name):
        channel = Channel.from_cache(self, name)
        return channel


@attr.s(auto_attribs=True)        # noqa: R0903
class ACE:
    permits: bool
    principal: str
    permission: str


@attr.s(kw_only=True, auto_attribs=True)        # noqa: R0903
class Channel(Cache):
    name: str = 'pypi'
    upstream_enabled: bool = True
    acl: typing.List[ACE] = [ACE(ACL.allow, ACL.unauthorized, ACL.read)]
    # TODO timeout goes to channel
    timeout: float = DEFAULT_CACHE_TIMEOUT

    @classmethod
    def from_path(cls, path, name):
        channel = Channel(root=path.joinpath(CHANNELS_NAME, name), name=name)
        return channel

    @classmethod
    def from_cache(cls, cache, name):
        channel = Channel(root=cache.path(CHANNELS_NAME, name), name=name)
        return channel

    def projects_path(self, *path):
        projects_path = self.path(PROJECTS_NAME, *path)
        return projects_path

    def releases_path(self, *path):
        releases_path = self.path(RELEASES_NAME, *path)
        return releases_path

    def project(self, project_name):
        project = Project(channel=self, name=project_name)
        return project

    async def update_project(self, data):
        """
        Create or update a projects metadata structures.

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

        await project.update_metadata(metadata)


@attr.s(kw_only=True, auto_attribs=True)        # noqa: R0903
class Info:
    author: str = attr.ib()
    author_email: str = attr.ib()
    bugtrack_url: str = attr.ib()
    classifiers: typing.List[str] = attr.ib()
    description: str = attr.ib()
    description_content_type: str = attr.ib()
    docs_url: typing.List[str] = attr.ib()
    download_url: str = attr.ib()
    downloads: typing.Dict[str, int] = attr.ib()
    home_page: str = attr.ib()
    keywords: str = attr.ib()
    license: str = attr.ib()
    maintainer: str = attr.ib()
    maintainer_email: str = attr.ib()
    name: str = attr.ib()
    package_url: str = attr.ib()
    platform: str = attr.ib()
    project_url: str = attr.ib()
    project_urls: typing.List[str] = attr.ib()
    release_url: str = attr.ib()
    requires_dist: typing.List[str] = attr.ib()
    requires_python: str = attr.ib()
    summary: str = attr.ib()
    version: semantic_version.Version = attr.ib(converter=ensure_semver)

    @classmethod
    def from_dict(cls, dct):
        info = cls(**dct)
        return info


@attr.s(kw_only=True, auto_attribs=True)
class Release:
    comment_text: str = attr.ib()
    digests: typing.Dict[str, str] = attr.ib()
    downloads: int = attr.ib()
    filename: str = attr.ib()
    has_sig: bool = attr.ib()
    md5_digest: str = attr.ib()
    packagetype: str = attr.ib()
    python_version: str = attr.ib()
    requires_python: str = attr.ib()
    size: int = attr.ib()
    upload_time: str = attr.ib()
    url: str = attr.ib()

    @utils.reify
    def upstream_url(self):
        upstream_url = self.url
        return upstream_url

    @utils.reify
    def file_name(self):
        parsed_upstream_url = urllib.parse.urlparse(self.upstream_url)
        file_name = pathlib.Path(parsed_upstream_url.path).name
        return file_name

    @classmethod
    def from_dict(cls, dct):
        release = cls(**dct)
        return release


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
        return attr.asdict(self)


@attr.s(kw_only=True, auto_attribs=True)
class Project:

    """A Project embodies a pypi project."""

    channel: Channel = attr.ib()
    name: str = attr.ib()

    @utils.reify
    def path(self):
        """The project root path."""
        path = self.channel.projects_path(self.name[0], self.name)
        return path

    @utils.reify
    def path_latest(self):
        """The directory of the latest version."""
        return self.path.joinpath(LATEST_NAME)

    @utils.reify
    def path_latest_info(self):
        """The info matadata for the latest version."""
        path = self.path_latest.joinpath(METADATA_FILENAME)
        return path

    @property
    def latest_version(self):
        """The semantic version number of the latest version."""
        if not self.path_latest.is_symlink():
            return None
        name = self.path_latest.resolve().name
        version = semantic_version.Version(name)
        return version

    @property
    def needs_update(self):
        """Either there is no latest version or the timeout is over."""
        try:
            return (time.time()
                    - self.path_latest.stat().st_mtime) >= self.timeout
        except FileNotFoundError:
            return True

    def version_path(self, version):
        """The path of a semantic version."""
        # we need to cast version nto str
        path = self.path.joinpath(str(version))
        return path

    def info_path(self, version):
        """The path to the metadata of a semantic version."""
        path = self.version_path(version).joinpath(METADATA_FILENAME)
        return path

    async def get_metadata(self, version=None):
        """Return the metadata for that project and version."""
        if version is None:
            version = self.latest_version

        async with aiofiles.open(self.info_path(version)) as f:
            data = await f.read()
            info = Info.from_dict(utils.json_loads(data))
        # contains all version releases
        releases = collections.defaultdict(list)
        # contans only releases of the selected version
        urls = []
        # find all <version>/files/*.json files
        for version_path in self.path.iterdir():
            name = version_path.name
            if name in ('latest',):
                continue
            found_version = semantic_version.Version(name)

            for path in version_path.glob('files/*.json'):
                async with aiofiles.open(path) as f:
                    data = await f.read()
                    release = Release.from_dict(utils.json_loads(data))
                    releases[found_version].append(release)
                    if found_version == version:
                        urls.append(release)

        metadata = Metadata(
            info=info,
            releases=releases,
            urls=urls
        )
        return metadata

    async def update_metadata(self, metadata):
        await self.update_info(metadata.info)
        await self.update_releases(metadata.releases)

    async def update_info(self, info):
        # lookup the latest info
        version_path = self.version_path(info.version)
        info_path = version_path.joinpath(METADATA_FILENAME)
        ensure_parents(info_path)
        async with aiofiles.open(info_path, 'x') as f:
            data = utils.json_dumps(attr.asdict(info))
            await f.write(data)

        if self.latest_version is None or info.version > self.latest_version:
            self.path_latest.symlink_to(
                version_path.relative_to(self.path_latest.parent)
            )

    async def update_releases(self, releases):
        for version, packages in releases.items():
            for release in packages:
                await self.update_release(version, release)

    async def update_release(self, version, release):
        """Update a package release."""
        version_path = self.version_path(version)
        release_metadata_path = version_path.joinpath(
            FILES_NAME, release.packagetype
        ).with_suffix('.json')
        ensure_parents(release_metadata_path)

        # write metadata
        async with aiofiles.open(release_metadata_path, 'x') as f:
            data = utils.json_dumps(attr.asdict(release,
                                                dict_factory=str_keyed_dict))
            await f.write(data)

        # link metadata to release path
        url_release_path_base = parse_url_path(release.url).parent
        release_path_metadata_link = self.channel.releases_path(
            url_release_path_base.relative_to('/'),
            METADATA_FILENAME
        )
        ensure_parents(release_path_metadata_link)
        symlink_relative_to(release_path_metadata_link, release_metadata_path)


@attr.s(kw_only=True, auto_attribs=True)
class UpstreamProject(Project):
    api_base: str = 'https://pypi.org/pypi'

    @utils.reify
    def url(self):
        def _gen():
            yield self.api_base
            yield self.name
            if self.version:
                yield self.version
            yield 'json'
        url = '/'.join(_gen())
        return url

    async def get_metadata(self):
        client_session = aiotask_context.get('client_session')
        async with client_session.get(self.url) as r:
            if r.status == 200:
                log.info('Loading metadata from upstream', url=self.url)
                metadata = await r.json()
                return Metadata.from_dict(metadata)

            if r.status == 404:
                log.error('Metadata not found', project=self)
                raise MetadataNotFound(self, r)

            log.error('Error while retrieving metadata', project=self)
            raise MetadataRetrievingError(self, r)


@attr.s(kw_only=True, auto_attribs=True)
class CachedProject(UpstreamProject):

    """A Project is a cached pypi project."""


    @utils.reify
    def metadata_upstream_path(self):
        """The cached version of upstream metadata."""
        name = ('pypi-metadata.json'
                if self.version is None
                else f'pypi-metadata-{self.version}.json')
        path = self.path.joinpath(name)
        return path

    @utils.reify
    def metadata_path(self):
        """Our transformed metadata.

        We change urls of releases here.
        """
        name = ('metadata.json'
                if self.version is None
                else f'metadata-{self.version}.json')
        path = self.path.joinpath(name)
        return path

    async def get_metadata(self):
        """Either lookup in cache or query pypi and store in cache."""
        # try to find actual versions if not modified since cache_timeout
        path = self.metadata_path
        if (path.is_file()
                and (time.time() - path.stat().st_mtime)
                < self.cache.timeout):
            # serve this file
            async with aiofiles.open(path) as f:
                log.info('Loading metadata from cache', path=path)
                data = utils.json_loads(await f.read())
                metadata = Metadata.from_dict(data)
        else:
            # create cache
            upstream_metadata = await super().get_metadata()
            async with aiofiles.open(
                    self.metadata_upstream_path, "w") as f:
                await f.write(utils.json_dumps(upstream_metadata))
            # transform upstream
            metadata = await self.transform(upstream_metadata)
            async with aiofiles.open(self.metadata_path, "w") as f:
                await f.write(utils.json_dumps(metadata))
        return metadata

    async def transform(self, upstream_metadata):
        """Inject our own release URL into upstream metadata."""
        async def _transform_releases(releases):
            releases = [
                await self.transform_url(release, prepare_cache=True)
                for release in releases
            ]
            return releases

        releases = {
            version: await _transform_releases(releases)
            for version, releases in upstream_metadata.releases.items()
        }
        urls = [
            await self.transform_url(release)
            for release in upstream_metadata.urls
        ]

        metadata = Metadata(
            info=upstream_metadata.info,
            last_serial=upstream_metadata.last_serial,
            releases=releases,
            urls=urls)
        return metadata

    async def transform_url(self, release, prepare_cache=False):
        """Transform the upstream url and prepare cache.

        Since pypi sends multiple equal urls in metadata, we do not want to
        always prepare the cache.

        """
        release_file = ReleaseFile.from_channel_url(channel=self.channel,
                                                    url=release.url,
                                                    metadata=release)
        if prepare_cache:
            # TODO prepare cache by examining cache_path for existing metadata
            await release_file.persist()

        transformed_url = await release_file.transform_url()

        return transformed_url


@attr.s(kw_only=True, auto_attribs=True)
class ReleaseFile:
    channel: Channel = attr.ib()
    path: str = attr.ib()
    metadata: Release = attr.ib(factory=dict)

    @classmethod
    def from_channel_url(cls, channel, url, metadata):
        """We assume that the url is unique over channelname and filename."""
        parsed_url = urllib.parse.urlparse(url)
        unique_name = ''.join((
            channel.name,
            pathlib.Path(parsed_url.path).name
        )).encode('utf-8')
        path_hashs = [
            hashlib.blake2s(unique_name,                       # noqa: E1101
                            digest_size=digest_size,
                            person=b'cacheReL').hexdigest()
            for digest_size in (1, 2, hashlib.blake2s.MAX_DIGEST_SIZE)  # noqa: E1101
        ]
        path = pathlib.Path().joinpath(*path_hashs)
        release_file = cls(channel=channel, path=path, metadata=metadata)
        return release_file

    @utils.reify
    def file_name(self):
        return self.path.name

    @utils.reify
    def cache_path(self):
        path = self.channel.cache.root.joinpath('releases')
        return path

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
        metadata = attr.evolve(self.metadata)
        request = aiotask_context.get('request')

        metadata.url = request.app.router['release'].url_for(
            channel_name=self.channel.name,
            path=str(self.path),
            file=self.file_name
        )
        return metadata

    def metadata_path(self, path):
        metadata_path = self.cache_path.joinpath(path, 'metadata.json')
        return metadata_path

    async def persist(self):
        # copy url metadata to cache path
        metadata_path = self.metadata_path(self.path)
        if not metadata_path.parent.is_dir():
            metadata_path.parent.mkdir(parents=True)

        async with aiofiles.open(metadata_path, 'w') as f:
            data = utils.json_dumps(attr.asdict(self.metadata))
            await f.write(data)
            log.info('Cached release metadata', path=metadata_path)

    @classmethod
    async def from_cached_metadata(cls, cache, path):
        cached_release = cls(cache=cache)
        metadata_path = cached_release.metadata_path(path)
        async with aiofiles.open(metadata_path) as f:
            data = await f.read()
            cached_release.metadata = utils.json_loads(data)
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

        watcher = inotipy.Watcher.create()
        watcher.watch(str(self.file_path_preparing),
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
