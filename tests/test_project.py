import pytest


@pytest.fixture
def tempdir(tmpdir):
    import pathlib
    yield pathlib.Path(tmpdir)


@pytest.fixture
def cache(tempdir):
    from pypare.pypi import model
    cache = model.Cache(root=tempdir)
    return cache


@pytest.fixture
def channel(cache):
    channel = cache.channel('test')
    return channel


async def test_create_project_by_upstream_metadata(channel, test_project_data, tempdir):
    import pathlib
    import aiofiles
    import json
    await channel.update_project(test_project_data)
    target_file_path = tempdir.joinpath('channels/test/projects/f/foobar/0.1.0/metadata.json')

    async with aiofiles.open(target_file_path) as f:
        info = json.loads(await f.read())
        assert info == test_project_data['info']

    project = channel.project('foobar')
    foobar_metadata = await project.get_metadata()


def test_get_metadata(channel):
    pass
