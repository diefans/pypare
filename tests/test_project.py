import pathlib

import pytest

TEST_DATA_BASE = pathlib.Path(__file__).parent.joinpath('data')


@pytest.fixture
def pypi_app(pypare_config):
    return pypare_config['pypi_app']


@pytest.fixture
def pypi_releases_route(pypi_app):
    from pypare import pypi
    releases_route = pypi_app.router[pypi.RELEASES_ROUTE]
    return releases_route


@pytest.fixture
def cache(tmpdir):
    from pypare.pypi import model
    cache = model.Cache(root=pathlib.Path(tmpdir))
    return cache


@pytest.fixture
def channel(cache, pypare_app, pypi_releases_route):     # noqa: W0613
    channel = cache.channel(
        'test',
        releases_route=pypi_releases_route
    )
    return channel


async def test_create_project_by_upstream_metadata(channel, test_project_data,
                                                   tmpdir):
    from pypare.pypi import model
    import attr
    import aiofiles
    import json
    await channel.store_project(test_project_data)
    target_file_path = pathlib.Path(tmpdir)\
        .joinpath('channels/test/projects/f/foobar/0.1.0/metadata.json')

    async with aiofiles.open(target_file_path) as f:
        info = json.loads(await f.read())
        assert info == test_project_data['info']

    project = channel.project('foobar')
    foobar_metadata = await project.get_metadata()
    test_metadata = model.Metadata.from_dict(test_project_data)

    assert test_metadata == foobar_metadata


@pytest.mark.datafiles(TEST_DATA_BASE.joinpath('cache'))
async def test_release_file_path(datafiles, channel,
                                 pypi_releases_route):
    from pypare.pypi import model

    cache = model.Cache(root=pathlib.Path(datafiles))
    channel = cache.channel(
        'test',
        releases_route=pypi_releases_route
    )
    release_url = pathlib.Path(
        '/test/releases/92/f89b/'
        'a34e48001939d59cdd421abd04218990f0b7d311b34c257acf2f6ded05c16148/'
        'foobar-0.1.0.txt'
    )
    relative_release_url = release_url.relative_to('/test/releases')

    release_file = await model.ReleaseFile.from_channel_release_url(
        channel,
        relative_release_url
    )
    assert release_file.path == channel.releases_path(
        model.parse_url_path(release_file.release.url)
        .relative_to('/test/releases')
    )


async def test_get_foobar_metadata(aiohttp_client, pypare_app):
    client = await aiohttp_client(pypare_app)

    r = await client.get('/pypi/test/foobar/json')
    data = await r.json()
    breakpoint()

