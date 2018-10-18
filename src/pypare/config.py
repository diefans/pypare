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
import functools
import importlib
import inspect
import itertools
import sys

import aiohttp.web
import structlog

from . import logging


class Config(dict):
    defaults = {}
    """Some config defaults."""

    sanitizer = {}
    """Some converter to sanitize config values."""

    def __init__(self, config=None, **kwargs):
        if config is None:
            config = {}

        sanitizer = self.sanitizer
        sane_config = (
            (k, sanitizer[k](v) if k in sanitizer else v)
            for k, v in itertools.chain(config.items(), kwargs.items())
        )

        super().__init__(sane_config)
        self.set_defaults(self.defaults)
        self._loaded_plugins = set()

    def set_defaults(self, config):
        """Applies default values from config.

        :param config: another :py:obj:`dict`
        """
        for key, value in config.items():
            self.setdefault(key, value)


class AioHttpConfig(Config):
    defaults = {
        'host': '0.0.0.0',
        'port': 8080,
        'plugins_function_name': 'plug_me_in',
        'plugins': [],
        'debug': True,
    }
    """Some config defaults."""

    sanitizer = {
        'plugins': list,
        'port': int,
        'debug': bool
    }
    """Some converter to sanitize config values."""

    def create_app(self, loop=None):
        """Create the aiohttp app."""
        app = aiohttp.web.Application(loop=loop)
        app._debug = self['debug']
        # we hook in our self
        app['config'] = self
        app.config = self
        app.plugin = functools.partial(self.plugin, app)
        return app

    def prepare_app(self, app):
        self.hook_plugins(app)

    async def plugin(self, app, name):
        plugin = resolve_dotted_name(name)

        if inspect.ismodule(plugin):
            # apply default name
            plugin = getattr(plugin, self.get('plugins_function_name'))

        if not inspect.iscoroutinefunction(plugin):
            raise ValueError(f'{plugin.__name__} must be an async function.')

        log = structlog.get_logger()
        if plugin in self._loaded_plugins:
            log.warning('Plugin already loaded',
                        plugin=f'{plugin.__module__}.{plugin.__qualname__}')
            return

        try:
            return await plugin(app)
        finally:
            log.info('Loaded plugin',
                     plugin=f'{plugin.__module__}.{plugin.__qualname__}')
            self._loaded_plugins.add(plugin)

    def hook_plugins(self, app):
        """Actually load all async plugins."""
        async def _apply_plugins():
            for name in self['plugins']:
                await self.plugin(app, name)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_apply_plugins())

    def run(self):
        app = self.create_app()
        self.prepare_app(app)

        log = structlog.get_logger()
        log.info('Starting aiohttp', host=self['host'], port=self['port'])
        aiohttp.web.run_app(
            app,
            host=self['host'],
            port=self['port'],
            access_log_class=logging.AccessLogger,
            print=None
        )


def resolve_dotted_name(name):
    """Use pkg_resources style dotted name to resolve a name.

    Modules are cached in sys.modules
    """
    part = ':'
    module_name, _, attr_name = name.partition(part)
    if part in attr_name:
        raise ValueError(f'Invalid name: {name}')

    if module_name in sys.modules:
        resolved = sys.modules[module_name]
    else:
        spec = importlib.util.find_spec(module_name)
        if not spec:
            raise ImportError(f'Module `{module_name}` has no spec.')
        resolved = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resolved)
        sys.modules[resolved.__name__] = resolved

    if attr_name:
        resolved = getattr(resolved, attr_name)

    return resolved
