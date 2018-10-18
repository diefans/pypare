import importlib

import pytest


# find a module without importing it
has_tokio = bool(importlib.util.find_spec('tokio'))
has_uvloop = bool(importlib.util.find_spec('uvloop'))


@pytest.fixture
def cli_runner():
    from click.testing import CliRunner
    return CliRunner()


# a single asyncio param is not possible
# https://github.com/pytest-dev/pytest-asyncio/issues/100
@pytest.mark.parametrize('loop_policy, _', [
    ('asyncio', ...),
    pytest.param('uvloop', ...,
                 marks=[pytest.mark.xfail]
                 if not has_uvloop else []),
    pytest.param('tokio', ...,
                 marks=[pytest.mark.xfail]
                 if not has_tokio else []),
])
def test_pypi_defaults(cli_runner, mocker, loop_policy, _):
    import asyncio
    from pathlib import Path
    from pypare import scripts

    mock_config = mocker.patch.object(scripts.config, 'AioHttpConfig')

    result = cli_runner.invoke(scripts.cli,
                               ['--loop', loop_policy, 'pypi'],
                               obj={}, auto_envvar_prefix='PYPARE')
    assert result.exit_code == 0

    mock_config.assert_called_with({
        'host': '0.0.0.0',
        'port': 3141,
        'plugins': (),
        'cache_root': Path('~').expanduser() / '.cache' / 'pypare',
        'base_path': '/pypi',
        'upstream_channel_name': 'pypi',
        'upstream_channel_api_base': 'https://pypi.org/pypi',
        'upstream_channel_timeout': 60 * 60 * 24,
    }, debug=False)
    mock_config.return_value.__getitem__.assert_called_with('plugins')
    mock_config.return_value.__getitem__.return_value.append\
        .assert_called_with('pypare.pypi')

    asyncio.set_event_loop_policy(None)
