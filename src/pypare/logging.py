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

import logging
import logging.config
import sys

import aiohttp
import structlog


LOGGING_LEVEL_NAMES = list(map(logging.getLevelName, sorted((
    logging.NOTSET, logging.DEBUG, logging.INFO,
    logging.WARN, logging.ERROR, logging.CRITICAL,
))))
DEFAULT_LOGGING_LEVEL = logging.getLevelName(logging.INFO)
CONTEXT_SETTINGS = dict(token_normalize_func=lambda x: x.upper())


def is_debug(level):
    return level == logging.getLevelName(logging.DEBUG)


class StdioToLog:

    """Delegate sys.stdout to a logger."""

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


class ExtractLogExtra:      # noqa: R0903

    """Extract log record attributes to structlog event_dict."""

    def __init__(self, *attrs):
        self.attrs = attrs

    def __call__(self, logger, method_name, event_dict):
        """
        Add the logger name to the event dict.
        """
        record = event_dict.get("_record")
        for attr_name in self.attrs:
            if hasattr(record, attr_name):
                attr = getattr(record, attr_name)
                event_dict[attr_name] = attr
        return event_dict


def setup_logging(*, tty=sys.stdout.isatty(),
                  level=logging.DEBUG, capture_warnings=True,
                  redirect_print=False, json_indent=None):
    """Set up structured logging for logstash.

    :param tty: if True renders colored logs
    :param level: the log level to be applied
    :param capture_warnings: redirect warnings to the logger
    :param redirect_print: use the logger to redirect printed messages
    """
    # normalize level
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    renderer = (structlog.dev.ConsoleRenderer()
                if tty else
                structlog.processors.JSONRenderer(indent=json_indent,
                                                  sort_keys=True)
                )
    timestamper = structlog.processors.TimeStamper(fmt="ISO", utc=True)
    pre_chain = [
        # Add the log level and a timestamp to the event_dict if the log entry
        # is not from structlog.
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.format_exc_info,
        ExtractLogExtra('spec', 'url', 'mimetype', 'has_body', 'swagger_yaml',
                        'method', 'path', 'operation_id', 'data'),
        timestamper,
    ]

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": renderer,
                "foreign_pre_chain": pre_chain,
            },
            "colored": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": renderer,
                "foreign_pre_chain": pre_chain,
            },
        },
        "handlers": {
            "default": {
                # "level": "DEBUG",
                "class": "logging.StreamHandler",
                "formatter": "colored",
            },
            # "file": {
            #     # "level": "DEBUG",
            #     "class": "logging.handlers.WatchedFileHandler",
            #     "filename": "test.log",
            #     "formatter": "plain",
            # },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": level,
                "propagate": True,
            },
        }
    })
    logging.captureWarnings(capture_warnings)
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    structlog.configure(
        processors=processors,
        context_class=structlog.threadlocal.wrap_dict(dict),
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if redirect_print:
        # redirect stdio print
        print_log = structlog.get_logger('print')
        sys.stderr = StdioToLog(print_log)
        sys.stdout = StdioToLog(print_log)

    # log uncaught exceptions
    sys.excepthook = uncaught_exception


def uncaught_exception(ex_type, ex_value, tb):  # noqa: C0103
    log_ = structlog.get_logger('sys.excepthook')
    log_.critical(event='uncaught exception', exc_info=(ex_type, ex_value, tb))


def merge_override_maps(maps):
    """Merge all maps and remove emptied keys."""
    merged = {}
    for m in maps:
        merged.update(m)

    remove_keys = {k for k, v in merged.items() if not v}

    for k in remove_keys:
        del merged[k]

    return merged


class AccessLogger(aiohttp.abc.AbstractAccessLogger):   # noqa: R0903
    def log(self, request, response, time):
        log = structlog.get_logger()
        log.info('Access',
                 remote=request.remote,
                 method=request.method,
                 path=request.path,
                 time=time,
                 status=response.status)
