"""Metrics reporting utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MetricsReporter(ABC):
    @abstractmethod
    def report(self, name: str, payload: Dict[str, Any]) -> None:
        ...


class LoggingMetricsReporter(MetricsReporter):
    def report(self, name: str, payload: Dict[str, Any]) -> None:
        logger.info("[metrics] %s: %s", name, payload)


__all__ = ["MetricsReporter", "LoggingMetricsReporter"]
