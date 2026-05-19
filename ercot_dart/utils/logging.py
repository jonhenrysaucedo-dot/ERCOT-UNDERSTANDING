"""
Structured JSON logging for the ERCOT DART pipeline.

Each log record emits a JSON line with: timestamp, level, module, message,
and any extra kwargs passed to the logger. This makes logs trivially
ingestible by Splunk, Datadog, or CloudWatch.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        # Surface any extra kwargs passed via logger.info("...", extra={...})
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a JSON-structured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
