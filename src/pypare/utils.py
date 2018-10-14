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

import functools
import json
import pathlib

import aiohttp.web
from packaging.version import Version
import structlog
import yarl

json_adapter_registry = {}
"""A mapping of types to json adapter functions."""


def json_adapter(json_cls, adapter=None):
    """Register an adapter function to transform an obj into something json
    serializable."""
    if adapter is None:
        return lambda func: json_adapter(json_cls, func)

    if json_cls not in json_adapter_registry:
        json_adapter_registry[json_cls] = adapter
    else:
        log = structlog.get_logger()
        log.warning('JSON adapter for this type already registered',
                    type=json_cls, adapter=json_adapter_registry[json_cls])

    return adapter


def find_json_adapter(json_obj):
    """Find the appropriate adapter based on the MRO of the instance type."""
    for json_obj_cls in json_obj.__class__.__mro__:
        adapter = json_adapter_registry.get(json_obj_cls)
        return adapter


def json_default_handler(obj):
    """Try to convert an object to a json renderable type.

    If that object has a ``__json__`` method we call it first.
    After all we try to adapt that result via registered json adapters.
    """
    if hasattr(obj, '__json__') and callable(obj.__json__):
        result = obj.__json__()
    else:
        result = obj

    # try to adapt
    adapter = find_json_adapter(obj)
    if adapter:
        result = adapter(obj)

    return result


json_loads = json.loads
json_dumps = functools.partial(json.dumps, default=json_default_handler)
json_response = functools.partial(aiohttp.web.json_response, dumps=json_dumps)


@json_adapter(pathlib.Path)
def adapt_path(obj):
    return str(obj)


@json_adapter(yarl.URL)
def adapt_url(obj):
    return str(obj)


@json_adapter(Version)
def adapt_version(obj):
    return str(obj)


class reify:        # noqa: R0903
    def __init__(self, wrapped):
        self.wrapped = wrapped
        functools.update_wrapper(self, wrapped)

    def __get__(self, inst, cls=None):
        if inst is None:
            return self
        val = self.wrapped(inst)
        setattr(inst, self.wrapped.__name__, val)
        return val
