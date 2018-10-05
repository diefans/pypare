pypare
======

A very simple pypi cache.

features
^^^^^^^^

- uses `aiohttp`_, `aiofiles`_, `inotipy`_

- queries metadata via pypi JSON API

- filesystem is the database

- serve releases while downloading


.. _`aiohttp`: http://aiohttp.readthedocs.io/
.. _`aiofiles`: https://pypi.org/project/aiofiles/
.. _`inotipy`: https://github.com/ldo/inotipy

todo
^^^^

- private channels with user, groups and permissions

- use `python-libaio`_ for file stuff

- nice ui

.. _`python-libaio`: https://github.com/vpelletier/python-libaio


running the cache
^^^^^^^^^^^^^^^^^

.. code-block::

    # pypare --help
    Usage: pypare [OPTIONS] COMMAND [ARGS]...

    Options:
      --log-level [NOTSET|DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                      The logging level.  [default: INFO]
      --loop [asyncio|uvloop]         Use a different loop policy.  [default:
                                      asyncio]
      --version                       Show the version and exit.
      --help                          Show this message and exit.

    Commands:
      pypi  Run a simple pypi caching proxy.


.. code-block::

    # pypare pypi --help
    Usage: pypare pypi [OPTIONS]

      Run a simple pypi caching proxy.

    Options:
      -p, --port INTEGER          The port to run the server  [default: 3141]
      -h, --host TEXT             The server host IP.  [default: 0.0.0.0]
      -b, --base-path PATH        The base path for this application.  [default:
                                  /pypi]
      -c, --cache-root DIRECTORY  The cache directory, where files are stored.
                                  [default: ~/.cache/pypare]
      --cache-timeout INTEGER     The age of metatdata, when it will be refreshed.
                                  [default: 86400]
      --plugin LIST               A plugin in pkg_resources notation to load.
      --help                      Show this message and exit.

Run from virtual environment:

.. code-block:: bash

   pip install pypare

   pypare pypi --cache-dir /tmp/pypi-data


Run in docker:

.. code-block:: bash

   docker run -it diefans/pypare:latest pypi


Run as zipapp:

.. code-block:: bash

   shiv pypare -c pypare -o ~/.local/bin/pypare -p ~/.pyenv/versions/3.7.0/bin/python

   pypare pypi --cache-dir /tmp/pypi-data

