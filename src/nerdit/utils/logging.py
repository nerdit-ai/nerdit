"""Logging configuration for Nerdit."""

import contextlib
import logging
import logging.handlers
import os
import sys


def _open_0600(path: str, flags: int) -> int:
    """`open()` opener creating the file mode 0600."""
    return os.open(path, flags, 0o600)


class _SecureRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Create logs and rotated backups with mode 0600 to protect runtime details.

    Override _open because FileHandler has no opener argument; renamed backups
    inherit the original file's permissions.
    """

    def _open(self):  # type: ignore[no-untyped-def]
        stream = open(
            self.baseFilename,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            opener=_open_0600,
        )
        # The opener mode only applies at creation — a pre-existing file (e.g.
        # a legacy 0644 log on an upgraded install) keeps its old mode without
        # this. Re-runs on every rotation, matching the boot-log precedent.
        with contextlib.suppress(OSError):
            os.chmod(self.baseFilename, 0o600)
        return stream


def setup_logging(
    level: str = "info",
    log_file: str | None = None,
    max_bytes: int = 0,
    backup_count: int = 0,
) -> None:
    """Configure stderr logging and optional private rotating file output.

    Enable a 0600 file handler when log_file is set and max_bytes > 0. Attach it
    also to non-propagating uvicorn loggers so access logs reach the bounded file.
    max_bytes=0 disables file logging.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    # httpx logs every request URL at INFO ("HTTP Request: GET http://..."),
    # path and query included. On the hosted app path (P26, core/link/mux.py)
    # that URL is a third-party app's — magic links and signed tokens ride in
    # those segments — so the mux redacts it on its own log sites; this keeps
    # the client library from reintroducing the leak one logger down.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    datefmt = "%Y-%m-%d %H:%M:%S"

    if log_file and max_bytes > 0:
        formatter = logging.Formatter(fmt, datefmt=datefmt)
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        file_handler = _SecureRotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setFormatter(formatter)
        # handlers= and stream= are mutually exclusive; no force=True (it would
        # nuke pytest's handlers — basicConfig is a no-op on re-call, acceptable
        # since the daemon calls this once per process and restart re-execs).
        logging.basicConfig(level=numeric_level, handlers=[stream_handler, file_handler])
        for name in ("uvicorn", "uvicorn.access"):
            lg = logging.getLogger(name)
            # Idempotent: drop any handler a prior call attached, so repeated
            # setup_logging calls (tests, embeddings) never stack duplicate
            # writers or hold rotated-away fds open.
            for stale in [h for h in lg.handlers if isinstance(h, _SecureRotatingFileHandler)]:
                lg.removeHandler(stale)
                stale.close()
            lg.addHandler(file_handler)
    else:
        logging.basicConfig(
            level=numeric_level,
            format=fmt,
            datefmt=datefmt,
            stream=sys.stderr,
        )
