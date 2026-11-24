import logging
import os
import re
import sys
import time
import uuid

import structlog
from flask import Flask, g, got_request_exception, request

QUIET_PATHS = {"/metrics", "/health"}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

log = structlog.get_logger("immocast")


def log_level() -> int:
    level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO").upper())
    return level if isinstance(level, int) else logging.INFO


def configure_logging():
    level = log_level()
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout, force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=False,
    )


def _request_id() -> str:
    provided = request.headers.get("X-Request-ID", "")
    return provided if REQUEST_ID_PATTERN.match(provided) else uuid.uuid4().hex[:12]


def register_request_logging(app: Flask):
    @app.before_request
    def start_request():
        structlog.contextvars.clear_contextvars()
        g.request_id = _request_id()
        g.request_start = time.perf_counter()
        structlog.contextvars.bind_contextvars(request_id=g.request_id)

    @app.after_request
    def log_request(response):
        duration_ms = round((time.perf_counter() - g.get("request_start", time.perf_counter())) * 1000, 2)
        response.headers["X-Request-ID"] = g.get("request_id", "")
        fields = {
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        }
        if response.status_code >= 500:
            log.error("request", **fields)
        elif response.status_code >= 400:
            log.warning("request", **fields)
        elif request.path in QUIET_PATHS:
            log.debug("request", **fields)
        else:
            log.info("request", **fields)
        return response

    @app.teardown_request
    def clear_request(_exc):
        structlog.contextvars.clear_contextvars()

    def log_exception(_sender, exception, **_extra):
        log.error("request_failed", path=request.path, error=type(exception).__name__, exc_info=exception)

    got_request_exception.connect(log_exception, app, weak=False)
