from __future__ import annotations

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import (
    datetime,
    timezone,
)
from typing import Any

request_id_var: ContextVar[
    str | None
] = ContextVar(
    "request_id",
    default=None,
)

class JsonFormatter(logging.Formatter):
    """
    Structured JSON log formatter.

    Responsibilities:
    - structured logs
    - request correlation
    - subsystem tagging
    - exception serialization
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        payload: dict[str, Any] = {
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = request_id_var.get()

        if request_id:

            payload["request_id"] = request_id

        subsystem = getattr(
            record,
            "subsystem",
            None,
        )

        if subsystem:

            payload["subsystem"] = subsystem

        for key, value in (
            record.__dict__.items()
        ):

            if key.startswith("_"):
                continue

            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }:
                continue

            if key not in payload:

                payload[key] = value

        if record.exc_info:

            payload["exception"] = {
                "type": (
                    record.exc_info[
                        0
                    ].__name__
                ),
                "message": str(record.exc_info[1]),
                "traceback": (
                    traceback.format_exception(
                        *record.exc_info
                    )
                ),
            }

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )

def configure_logging(
    *,
    level: int = logging.INFO,
) -> None:
    """
    Global logging configuration.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(
        root_logger.handlers
    ):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)

def get_logger(
    name: str,
    *,
    subsystem: str | None = None,
) -> logging.LoggerAdapter:
    """
    Structured subsystem logger.
    """

    logger = logging.getLogger(name)

    return logging.LoggerAdapter(
        logger,
        extra={
            "subsystem": subsystem
        },
    )

def set_request_id(
    request_id: str,
) -> Any:
    """
    Binds request correlation ID.
    """

    return request_id_var.set(request_id)


def reset_request_id(
    token: Any,
) -> None:
    """
    Restores previous request correlation ID.
    """

    request_id_var.reset(token)

def clear_request_id() -> None:
    """
    Clears request-scoped ID.
    """

    request_id_var.set(None)

def log_exception(
    logger: (
        logging.Logger
        | logging.LoggerAdapter
    ),
    *,
    message: str,
    exc: Exception,
    **extra,
) -> None:
    """
    Standardized exception logging.
    """

    logger.error(
        message,
        exc_info=exc,
        extra={
            **extra,
            "exception_type": (
                type(exc).__name__
            ),
        },
    )

