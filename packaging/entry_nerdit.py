"""Launch the CLI as a PyInstaller script entry point.

Call freeze_support first so frozen children bootstrap workers instead of
restarting the CLI. Keep business logic in the tested CLI package.
"""

from __future__ import annotations

import multiprocessing


def _run() -> None:
    multiprocessing.freeze_support()

    from nerdit.cli.app import main

    main()


if __name__ == "__main__":
    _run()
