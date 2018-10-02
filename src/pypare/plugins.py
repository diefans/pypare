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

"""Common middlewares to enhance microservices."""
import asyncio

import aiohttp.web
import aiohttp.client
import aiotask_context as context


async def plugin_task_context(app):
    """Add actual config, request and a :py:obj:`ClientSession` to the task
    context."""
    loop = asyncio.get_event_loop()
    loop.set_task_factory(context.task_factory)
    client_session = aiohttp.client.ClientSession()

    async def close_client_session(app):    # noqa: W0613
        await client_session.close()
    app.on_cleanup.append(close_client_session)

    @aiohttp.web.middleware
    async def task_context_middleware(request, handler):        # noqa: W0613
        context.set('request', request)
        context.set('config', app.config)
        context.set('client_session', client_session)
        response = await handler(request)
        return response

    app.middlewares.append(task_context_middleware)
