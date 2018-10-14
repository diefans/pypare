import pathlib

import pytest


@pytest.fixture
def pypare_config(tmpdir):
    from pypare import config
    conf = config.AioHttpConfig({
        'base_path': '/pypi',
        'cache_root': pathlib.Path(tmpdir)
    })
    return conf


@pytest.fixture
def pypare_app(pypare_config, loop):
    # we need loop here to initialize
    pypare_config['plugins'].append('pypare.pypi')
    app = pypare_config.create_app()
    pypare_config.prepare_app(app)
    return app


@pytest.fixture(scope='session')
def test_data_dir():
    data_dir = pathlib.Path(__file__).parent.joinpath('data')
    return data_dir


@pytest.fixture
def test_project_data_json(test_data_dir):
    with test_data_dir.joinpath('project.json').open() as f:
        yield f.read()


@pytest.fixture
def test_project_data(test_project_data_json):
    import json
    yield json.loads(test_project_data_json)
