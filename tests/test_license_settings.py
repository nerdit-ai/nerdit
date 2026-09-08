"""The ``[license]`` config section (P17d D-LIC4).

One key — ``file`` — registered in the three places that make a section real:
``NerditSettings``, ``_SECTION_MODELS`` and ``_RESTART_KEYS`` (plus the
``load_settings`` TOML mapping). It then gets ``GET/PUT
/config/daemon/license`` + declarative apply + restart-drift detection for
free, with no new config surface.

The deliberate asymmetry that reviewers should see pinned here: the *path* is
restart-keyed because it is resolved once at boot, while the license *content*
is refreshed in place by the install route — no surface may assume boot-frozen
license state.

.. note:: The section points at the *product* license a customer installs,
   which is unrelated to the licensing of this repository (Apache-2.0).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nerdit.config.settings import LicenseSettings, NerditSettings, load_settings
from nerdit.config.store import ConfigError, ConfigStore

STORED_LICENSE = """
[license]
file = "/tmp/old-license.jws"
"""


def _store(tmp_path: Path, body: str = "") -> ConfigStore:
    path = tmp_path / "config.toml"
    if body:
        path.write_text(body)
    return ConfigStore(path)


# --- defaults / shape ---------------------------------------------------------


def test_defaults():
    license_settings = LicenseSettings()
    assert license_settings.file is None
    assert NerditSettings().license == license_settings


def test_exactly_one_key():
    """No grace knob (policy), no trusted keys (D-LIC3), no enable flag.

    An installed file *is* the enable, and its absence is the ordinary
    unlicensed state of the free local product.
    """
    assert set(LicenseSettings.model_fields) == {"file"}


# --- validator ----------------------------------------------------------------


def test_file_validation():
    assert LicenseSettings(file="/srv/nerdit/license.jws").file == "/srv/nerdit/license.jws"
    assert LicenseSettings(file="~/license.jws").file == "~/license.jws"
    assert LicenseSettings(file=None).file is None
    for bad in ("relative/license.jws", "  ", ""):
        with pytest.raises(ValidationError):
            LicenseSettings(file=bad)


def test_validator_never_interpolates_file_content():
    """The message may name the operator-authored path; nothing else exists yet.

    The validator runs on a *path string*, never on a blob — this pins that no
    future edit starts reading the file here (a 422 that echoed license bytes
    would be the P22 ``extra_forbidden`` lesson all over again).
    """
    with pytest.raises(ValidationError) as exc:
        LicenseSettings(file="not/absolute.jws")
    assert "not/absolute.jws" in str(exc.value)
    assert "<data_dir>/license.jws" in str(exc.value)


def test_unresolvable_tilde_user_is_a_value_error_not_a_500():
    with pytest.raises(ValidationError):
        LicenseSettings(file="~nosuchuser4242/license.jws")


@pytest.mark.parametrize("bad", ["/tmp/a\x00b.jws", "/tmp/a\nb.jws", "/tmp/a\x7fb.jws"])
def test_control_characters_in_the_path_are_refused(bad):
    """A NUL survives a TOML round-trip but wedges the NEXT boot.

    ``os.open`` raises ``ValueError`` — not ``OSError`` — on a path with an
    embedded NUL, a class ``build_license_state`` does not degrade on, so a
    config PUT that persisted one would abort the following daemon boot. The
    validator is the operator's moment to hear about it (structured 422); the
    read boundary in ``core/license.py`` is the second belt for a path already
    written by a pre-fix daemon.

    The offending value is deliberately never echoed back: control characters in
    an error string are their own hazard.
    """
    with pytest.raises(ValidationError) as exc:
        LicenseSettings(file=bad)
    assert "control characters" in str(exc.value)
    assert "\x00" not in str(exc.value)


# --- config-as-API plumbing ---------------------------------------------------


def test_section_registered(tmp_path):
    assert "license" in _store(tmp_path).known_sections()


def test_all_keys_restart_required(tmp_path):
    """The path is resolved once at boot and ``app.state.license`` is built
    from it, so a config PUT only rewrites TOML ([git]/[mcp]/[link] precedent)."""
    store = _store(tmp_path, STORED_LICENSE)
    body = {"file": "/srv/nerdit/license.jws"}
    staged = store.stage("license", body)

    assert staged.requires_restart is True
    assert all(entry.requires_restart for entry in staged.diff)
    assert {entry.key for entry in staged.diff} == {"license.file"}
    assert set(staged.restart_keys) == {"license.file"}
    # The frozenset covers the whole model, not just the key we happened to write.
    assert set(body) == set(LicenseSettings.model_fields)


def test_unknown_key_is_a_422(tmp_path):
    # There is no [license].trusted_keys override in v1 (D-LIC3): the config API
    # answers 422 rather than storing a knob nothing reads.
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("license", {"trusted_keys": {"kid": "ed25519:x"}})
    assert exc.value.status_code == 422
    assert exc.value.code == "config.invalid"
    assert [d.type for d in exc.value.diagnostics] == ["extra_forbidden"]


def test_stage_invalid_reports_diagnostics(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _store(tmp_path).stage("license", {"file": "relative.jws"})
    assert exc.value.status_code == 422
    assert ["file"] in [d.loc for d in exc.value.diagnostics]


def test_null_delete_reverts_to_the_data_dir_default(tmp_path):
    store = _store(tmp_path, STORED_LICENSE)
    store.commit(store.stage("license", {"file": None}))
    assert store.effective_section("license")["file"] is None


def test_load_settings_maps_license(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[license]\nfile = "/srv/nerdit/license.jws"\n')
    assert load_settings(path).license.file == "/srv/nerdit/license.jws"


def test_view_section_has_no_secret_material(tmp_path):
    """``file`` is a plain path leaf, exactly like ``security.secrets_key_file``.

    The blob itself never enters config: it lives in the file the path names.
    """
    store = _store(tmp_path, STORED_LICENSE)
    view = store.view_section("license")
    assert view == store.effective_section("license")
    assert view["file"] == "/tmp/old-license.jws"


def test_restart_key_asymmetry_is_documented(tmp_path):
    """The path is restart-keyed; the *content* deliberately is not.

    ``POST /api/license`` refreshes ``app.state.license`` in place, so doctor
    and ``/capabilities`` tell the truth without a restart. Pinned as source
    text because it is the sentence a reviewer needs to find.
    """
    source = Path("src/nerdit/config/store.py").read_text(encoding="utf-8")
    block = source.split('"license": frozenset')[0][-900:]
    assert "restart-keyed" in block
    assert "boot-frozen license state" in block
