"""Launch the daemon as a PyInstaller script entry point.

Call freeze_support first so frozen children bootstrap workers instead of
restarting the daemon. Keep business logic in the tested daemon package.
"""

from __future__ import annotations

import multiprocessing


def _run() -> None:
    multiprocessing.freeze_support()

    from nerdit.daemon.server import main

    main()


if __name__ == "__main__":
    _run()
