"""Structured logging + stage timing with peak-memory snapshots."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def peak_rss_mb() -> Optional[float]:
    """Peak resident memory in MB (Windows: peak_wset; POSIX: ru_maxrss)."""
    try:
        import psutil  # type: ignore
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows only
        if peak:
            return round(peak / (1024 ** 2), 1)
    except Exception:
        mi = None
    try:
        import resource  # type: ignore
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return round(ru / (1024 ** 2), 1)
        return round(ru / 1024.0, 1)
    except Exception:
        pass
    if mi is not None:  # fallback: current RSS
        return round(mi.rss / (1024 ** 2), 1)
    return None


class StageTimer:
    """Timer for a pipeline stage; logs duration and peak RSS on stop()."""

    def __init__(self, stage: str, cfg=None, logger: Optional[logging.Logger] = None):
        self.stage = stage
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.seconds: Optional[float] = None
        self.peak_rss_mb: Optional[float] = None
        self._t0: Optional[float] = None

    def start(self) -> "StageTimer":
        self._t0 = time.monotonic()
        self.logger.info("STAGE %s START", self.stage)
        return self

    def stop(self) -> "StageTimer":
        if self._t0 is not None:
            self.seconds = round(time.monotonic() - self._t0, 2)
        self.peak_rss_mb = peak_rss_mb()
        self.logger.info("STAGE %s DONE in %ss (peak RSS %s MB)",
                         self.stage, self.seconds, self.peak_rss_mb)
        return self

    def __enter__(self) -> "StageTimer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def stage_timer(stage: str, cfg=None) -> StageTimer:
    return StageTimer(stage, cfg)


def setup_logging(cfg=None) -> logging.Logger:
    """Configure root logging: console (+ file under the artifact dir when cfg given)."""
    root = logging.getLogger()
    level = "INFO"
    if cfg is not None:
        level = str(getattr(getattr(cfg, "logging", None), "level", "INFO")
                    or "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(sh)
    if cfg is not None:
        try:
            log_dir = Path(getattr(cfg.logging, "dir", "")
                           or (Path(cfg.paths.artifact_dir) / "logs")).expanduser()
            log_dir.mkdir(parents=True, exist_ok=True)
            if not any(getattr(h, "ber_pipeline_log", False) for h in root.handlers):
                fh = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
                fh.setFormatter(logging.Formatter(LOG_FORMAT))
                fh.ber_pipeline_log = True  # type: ignore[attr-defined]
                root.addHandler(fh)
        except Exception as exc:
            root.warning("file logging disabled (%s)", exc)
    return root
