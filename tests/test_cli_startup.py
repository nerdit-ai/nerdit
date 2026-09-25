"""The CLI must start without loading the daemon's server stack.

Every `nerdit` invocation imports `nerdit.cli.app`, which registers all
commands. FastAPI alone costs ~150 ms there, so a module-level import of a
daemon module that pulls it in slows down every command.
"""

from __future__ import annotations

import subprocess
import sys


def test_cli_app_import_does_not_load_fastapi() -> None:
    code = "import sys, nerdit.cli.app; print('fastapi' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True
    )
    assert result.stdout.strip() == "False"
