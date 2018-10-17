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
import importlib
from pathlib import Path

import click
import structlog

from .. import __version__, config, logging


def find_loop_specs():
    """Just find specs for common loops."""
    module_specs = (
        (module_name, importlib.util.find_spec(module_name))
        for module_name in ('asyncio', 'uvloop', 'tokio')
    )
    available_specs = {
        module_name: spec for module_name, spec in module_specs
        if spec is not None
    }
    return available_specs


def set_loop_policy(event_loop):
    log = structlog.get_logger()
    if event_loop == 'uvloop':
        try:
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            log.info("Using uvloop event loop policy")

        except ImportError:
            log.warning("uvloop is not available.")

    elif event_loop == 'tokio':
        try:
            import tokio
            asyncio.set_event_loop_policy(tokio.EventLoopPolicy())
            log.info("Using tokio event loop policy")

        except ImportError:
            log.warning("tokio is not available.")

    else:
        # set default policy
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    # loop = asyncio.get_event_loop()
    # return loop


@click.group(context_settings=logging.CONTEXT_SETTINGS)
@click.option('--log-level', default=logging.DEFAULT_LOGGING_LEVEL,
              show_default=True,
              type=click.Choice(logging.LOGGING_LEVEL_NAMES),
              help='The logging level.')
@click.option('event_loop', '--loop', default='asyncio',
              show_default=True,
              type=click.Choice(find_loop_specs().keys()),
              help='Use a different loop policy.')
@click.version_option(__version__)
@click.pass_obj
def cli(obj, log_level, event_loop):
    obj['debug'] = logging.is_debug(log_level)
    logging.setup_logging(level=log_level)
    set_loop_policy(event_loop)


class ClickPath(click.Path):
    def convert(self, value, param, ctx):
        value = super().convert(value, param, ctx)
        value = Path(value).expanduser().resolve()
        return value


@cli.command('pypi')
@click.option('-p', '--port', default=3141, type=int,
              show_default=True,
              help='The port to run the server')
@click.option('-h', '--host', default='0.0.0.0',
              show_default=True,
              help='The server host IP.')
@click.option('base_path', '-b', '--base-path', type=click.Path(),
              default='/pypi', show_default=True,
              help='The base path for this application.')
@click.option('cache_root', '-c', '--cache-root',
              type=ClickPath(file_okay=False, dir_okay=True, writable=True),
              default='~/.cache/pypare', show_default=True,
              help='The cache directory, where files are stored.',
              )
@click.option('upstream_channel_name', '-u', '--upstream-channel',
              default='pypi', help='The name of the upstream channel.')
@click.option('upstream_channel_api_base', '--upstream-channel-url',
              default='https://pypi.org/pypi',
              help='The base API URL of the upstream channel.')
@click.option('upstream_channel_timeout', '--upstream-channel-timeout',
              default=60 * 60 * 24,
              help='The timeout upstream is asked for new metadata.')
@click.option('plugins', '--plugin', multiple=True, type=list,
              help='A plugin in pkg_resources notation to load.')
@click.pass_obj
def cli_cache(obj, **pypi_config):
    """Run a simple pypi caching proxy."""
    conf = config.AioHttpConfig(pypi_config, debug=obj['debug'])
    conf['plugins'].append('pypare.pypi')
    conf.run()


# @cli.command('resolve')
# def cli_resolve():
#     pass


def main():
    cli(obj={}, auto_envvar_prefix='PYPARE')    # noqa: E1123
