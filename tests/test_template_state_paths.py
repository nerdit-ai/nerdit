"""Every per-node path the daemon defaults to lives under ~/.nerdit.

`install.sh --template` makes a VM template fork-safe by deleting whole trees
(the `rm -rf "$UNIT_HOME/.nerdit" ...` line in packaging/install.sh), not a list
of files. That only holds while no default points outside ~/.nerdit, so a new
per-node default elsewhere must fail here and extend that line.
"""

from pathlib import Path, PurePath

from pydantic import BaseModel

from nerdit.config.settings import NerditSettings
from nerdit.core.license import resolve_license_file
from nerdit.core.link.identity import resolve_key_file
from nerdit.core.secrets import SecretManager


def _walk(model: BaseModel, prefix: str = ""):
    for name in type(model).model_fields:
        value = getattr(model, name)
        value = str(value) if isinstance(value, PurePath) else value
        if isinstance(value, BaseModel):
            yield from _walk(value, f"{prefix}{name}.")
        else:
            yield f"{prefix}{name}", name, value


def test_every_default_path_stays_under_the_wiped_tree():
    seen = set()
    for dotted, name, value in _walk(NerditSettings()):
        is_path_field = name in ("data_dir", "file") or name.endswith(("_file", "_dir"))
        if isinstance(value, str) and value and (is_path_field or value.startswith("~")):
            assert value.startswith("~/.nerdit"), f"{dotted} = {value!r}"
            seen.add(dotted)
    # Not vacuous: the walk reaches nested sections.
    assert {"data_dir", "daemon.pid_file", "daemon.upload_dir"} <= seen


def test_derived_defaults_stay_under_the_data_dir():
    data_dir = NerditSettings().data_dir
    assert data_dir == "~/.nerdit"
    root = Path(data_dir).expanduser()
    assert resolve_key_file(None, data_dir).is_relative_to(root)
    assert resolve_license_file(None, data_dir).is_relative_to(root)
    # Built the way daemon/bootstrap.py builds it.
    assert SecretManager(root / "secrets").key_path.is_relative_to(root)
