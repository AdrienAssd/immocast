import threading
from collections import deque

import pandas as pd
import structlog

from ml.drift import drift_report

log = structlog.get_logger("immocast.drift")


class DriftMonitor:
    """Keeps the last `window` inputs and recomputes drift against the champion's training profile
    every `every` predictions."""

    def __init__(self, gauge, window: int = 500, every: int = 100):
        self.gauge = gauge
        self.window = deque(maxlen=window)
        self.every = every
        self.seen = 0
        self.profile = None
        self.lock = threading.Lock()

    def reset(self, profile: dict | None):
        with self.lock:
            self.profile = profile
            self.window.clear()
            self.seen = 0
            self.gauge.clear()

    def observe(self, features: dict) -> dict | None:
        with self.lock:
            if self.profile is None:
                return None
            self.window.append(features)
            self.seen += 1
            if self.seen % self.every:
                return None
            report = drift_report(self.profile, pd.DataFrame(list(self.window)))
            for feature, value in report.items():
                self.gauge.labels(feature=feature).set(value)
        log.info("drift_computed", rows=len(self.window), **{f"drift_{k}": v for k, v in report.items()})
        return report
