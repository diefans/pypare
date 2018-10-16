from pathlib import Path

import asynctest
import pytest

TEST_DATA_BASE = Path(__file__).parent / 'data'
pytestmark = pytest.mark.datafiles(TEST_DATA_BASE / 'cache')


@pytest.fixture
def pypare_config(datafiles):
    from pypare import config
    conf = config.AioHttpConfig({
        'base_path': '/pypi',
        'cache_root': Path(datafiles)
    })
    return conf


@pytest.fixture
def pypare_app(pypare_config, loop):        # noqa: W0613
    # we need loop here to initialize
    pypare_config['plugins'].append('pypare.pypi')
    app = pypare_config.create_app()
    pypare_config.prepare_app(app)
    return app


@pytest.fixture
def pypi_app(pypare_config, pypare_app):    # noqa: W0613
    return pypare_config['pypi_app']


@pytest.fixture
def pypi_releases_route(pypi_app):
    from pypare import pypi
    releases_route = pypi_app.router[pypi.RELEASES_ROUTE]
    return releases_route


@pytest.fixture
def cache(datafiles):
    from pypare.pypi import model
    cache = model.Cache(root=Path(datafiles))
    return cache


@pytest.fixture
async def channel(cache, pypi_releases_route):
    channel = await cache.channel(
        'test',
        releases_route=pypi_releases_route
    )
    return channel


@pytest.fixture
def jsondata():
    import aiofiles
    from pypare import utils

    async def _loader(*path):
        async with aiofiles.open(TEST_DATA_BASE / Path(*path)) as f:
            data = utils.json_loads(await f.read())
            return data

    return _loader


async def test_create_project_by_upstream_metadata(jsondata,
                                                   channel, datafiles):
    from pypare.pypi import model
    import aiofiles
    import json
    test_project_data = await jsondata('project.json')
    await channel.store_project(test_project_data)
    target_file_path = Path(datafiles)\
        / 'channels/test/projects/b/bimbam/0.1.0/metadata.json'

    async with aiofiles.open(target_file_path) as f:
        info = json.loads(await f.read())
        assert info == test_project_data['info']

    project = channel.project('bimbam')
    project_metadata = await project.load_metadata()
    test_metadata = model.Metadata.from_dict(test_project_data)

    assert test_metadata == project_metadata


async def test_get_foobar_metadata(aiohttp_client, pypare_app, jsondata):
    client = await aiohttp_client(pypare_app)

    r = await client.get('/pypi/test/foobar/json')
    data = await r.json()
    test_project_data = await jsondata('project-2.json')
    assert test_project_data == data


async def test_get_foobar_release_stream(aiohttp_client, pypare_app):
    client = await aiohttp_client(pypare_app)

    r = await client.get(
        '/pypi/test/+releases/92/f89b/'
        'a34e48001939d59cdd421abd04218990f0b7d311b34c257acf2f6ded05c16148/'
        'foobar-0.1.0.txt')

    buf = bytearray()
    assert r.status == 200
    async for data, _ in r.content.iter_chunks():
        buf.extend(data)

    assert buf == bytearray(b'0123456789\n')


async def test_upstream_get_foobar_metadata(aiohttp_client, pypare_app,
                                            mocker,
                                            jsondata):
    test_project_data = await jsondata('project-2.json')
    mock_session = mocker.patch(
        'pypare.pypi.model.aiotask_context.get').return_value
    mock_response = asynctest.MagicMock(name='response')
    mock_session.get.return_value = mock_response

    mock_response.__aenter__.return_value.status = 200
    mock_json = asynctest.CoroutineMock()
    mock_response.__aenter__.return_value.json = mock_json
    mock_json.return_value = test_project_data

    client = await aiohttp_client(pypare_app)

    r = await client.get('/pypi/test-upstream/foobar/json')
    data = await r.json()
    test_project_data = await jsondata('project-2-upstream.json')

    assert test_project_data == data


async def test_upstream_get_foobar_release_stream(aiohttp_client, pypare_app,
                                                  mocker):
    mock_session = mocker.patch(
        'pypare.pypi.model.aiotask_context.get').return_value

    mock_head = asynctest.MagicMock(name='head')
    mock_head.__aenter__.return_value.status = 200
    mock_session.head.return_value = mock_head

    mock_iter_chunks = asynctest.MagicMock(name='iter_chunks')
    mock_iter_chunks.__aiter__.return_value = iter([
        (b'0', ...),
        (b'1', ...),
        (b'2', ...),
        (b'3', ...),
        (b'4', ...),
        (b'5', ...),
        (b'6', ...),
        (b'7', ...),
        (b'8', ...),
        (b'9', ...),
        (b'\n', ...),
    ])

    mock_response = asynctest.MagicMock(name='response')
    mock_response.__aenter__.return_value.status = 200
    mock_response.__aenter__.return_value.content.iter_chunks.return_value \
        = mock_iter_chunks
    mock_session.get.return_value = mock_response

    client = await aiohttp_client(pypare_app)

    r = await client.get(
        '/pypi/test-upstream/+releases/92/f89b/'
        'a34e48001939d59cdd421abd04218990f0b7d311b34c257acf2f6ded05c16148/'
        'foobar-0.1.0.txt')

    buf = bytearray()
    assert r.status == 200
    async for data, _ in r.content.iter_chunks():
        buf.extend(data)

    assert buf == bytearray(b'0123456789\n')
