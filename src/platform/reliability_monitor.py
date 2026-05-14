from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

APP_STARTED_AT = time.time()
REQUEST_EVENTS: deque[dict[str, Any]] = deque(maxlen=1000)


def configure_logging() -> None:
    root = logging.getLogger("emergeai")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    app_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)

    error_handler = logging.FileHandler(LOG_DIR / "error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    root.addHandler(app_handler)
    root.addHandler(error_handler)


def app_logger() -> logging.Logger:
    configure_logging()
    return logging.getLogger("emergeai.app")


def error_logger() -> logging.Logger:
    configure_logging()
    return logging.getLogger("emergeai.error")


def record_request(path: str, method: str, status_code: int, latency_ms: float) -> None:
    REQUEST_EVENTS.append(
        {
            "path": path,
            "method": method,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
        }
    )


def uptime_seconds() -> float:
    return round(time.time() - APP_STARTED_AT, 2)


def request_metrics() -> dict[str, Any]:
    events = list(REQUEST_EVENTS)
    latencies = [event["latency_ms"] for event in events]
    failed = [event for event in events if event["status_code"] >= 500]
    return {
        "uptime_seconds": uptime_seconds(),
        "request_count_window": len(events),
        "average_api_latency_ms": round(mean(latencies), 2) if latencies else None,
        "max_api_latency_ms": round(max(latencies), 2) if latencies else None,
        "failed_requests": len(failed),
        "recent_failed_requests": failed[-10:],
    }
