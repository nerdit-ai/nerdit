"""Tests for nerdit.toml project configuration parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from nerdit.config.project import find_project_config, load_project_config

# --- find_project_config ---


def test_find_project_config_in_cwd(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[run]\nscript = 'train.py'\n")
    result = find_project_config(start=tmp_path)
    assert result == toml_file


def test_find_project_config_walks_up(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[run]\nscript = 'train.py'\n")
    subdir = tmp_path / "src" / "models"
    subdir.mkdir(parents=True)
    result = find_project_config(start=subdir)
    assert result == toml_file


def test_find_project_config_not_found(tmp_path):
    subdir = tmp_path / "empty"
    subdir.mkdir()
    result = find_project_config(start=subdir)
    assert result is None


# --- load_project_config -----------------------------------------------------


def test_load_project_config_no_file():
    config = load_project_config(path=Path("/nonexistent/nerdit.toml"))
    assert config is None


def test_load_project_config_ignores_leftover_batch_sections(tmp_path):
    """A pre-pivot repo still carrying [run]/[data]/[scheduling] must still parse.

    WP5 dropped ``RunConfig``/``SchedulingConfig``/``ProjectConfig.data``.
    ``load_project_config`` builds the model from explicit per-section kwargs,
    so a stale table is simply never read — ``nerdit deploy`` must NOT start
    failing at parse time for repos that never cleaned their nerdit.toml.
    """
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text(
        '[run]\nscript = "train.py"\ngpus = 2\n\n'
        '[data]\ndataset = "/data/imagenet:/data"\n\n'
        '[scheduling]\ntime_window = "22:00-06:00"\n\n'
        "[deploy]\nname = 'my-app'\nport = 8000\n"
    )

    config = load_project_config(path=toml_file)

    assert config is not None
    assert config.deploy is not None
    assert config.deploy.name == "my-app"
    assert not hasattr(config, "run")
    assert not hasattr(config, "data")
    assert not hasattr(config, "scheduling")


# --- [deploy] section (P2 / S4) ----------------------------------------------


def test_deploy_config_parsed(tmp_path):
    from nerdit.config.project import load_project_config

    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[deploy]\nname = 'my-app'\nport = 8000\ngpus = 0\nstart = 'npm start'\n")
    config = load_project_config(path=toml_file)
    assert config.deploy is not None
    assert config.deploy.name == "my-app"
    assert config.deploy.port == 8000
    assert config.deploy.gpus == 0
    assert config.deploy.start == "npm start"
    assert config.deploy.health is None


def test_deploy_absent_yields_none(tmp_path):
    from nerdit.config.project import load_project_config

    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text('[run]\nscript = "train.py"\n')
    config = load_project_config(path=toml_file)
    assert config.deploy is None


def test_deploy_gpus_default_zero(tmp_path):
    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="svc", port=9000)
    assert cfg.gpus == 0
    assert cfg.start is None


# --- [deploy] validation (P4 / S3) --------------------------------------------


def test_deploy_config_valid():
    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="my-app-2", port=8000, gpus=1)
    assert cfg.name == "my-app-2"
    assert cfg.port == 8000
    assert cfg.gpus == 1


@pytest.mark.parametrize("bad_name", ["My_App", "-bad", "UPPER", "bad-", "", "a" * 64])
def test_deploy_config_invalid_name(bad_name):
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError, match="DNS label"):
        DeployConfig(name=bad_name, port=8000)


def test_deploy_config_volumes_valid():
    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="my-app", volumes=["data:/data", "cache:/var/cache"])
    assert cfg.volumes == ["data:/data", "cache:/var/cache"]


@pytest.mark.parametrize(
    "bad_volumes",
    [
        ["data"],  # missing ':'
        ["Bad:/data"],  # uppercase volname
        ["under_score:/data"],  # underscore volname
        ["data:relative"],  # non-absolute container path
        ["data:/"],  # forbidden root
        ["data:/workspace"],  # forbidden workspace
        ["data:/proc"],  # kernel-virtual root
        ["data:/data/../etc"],  # non-normalized
        ["data:/a", "data:/b"],  # duplicate name
        ["one:/data", "two:/data"],  # duplicate path
        [f"v{i}:/p{i}" for i in range(9)],  # >8 volumes
        ["../../etc:/x"],  # traversal never encodes in a volname
        ["x://workspace"],  # doubled leading slash — kernel collapses //x -> /x
        ["x://"],  # doubled slash collapsing to forbidden root
        ["x://proc"],  # doubled slash collapsing to a kernel-virtual root
    ],
)
def test_deploy_config_volumes_invalid(bad_volumes):
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError):
        DeployConfig(name="my-app", volumes=bad_volumes)


@pytest.mark.parametrize("bad_port", [0, 70000, -1])
def test_deploy_config_port_out_of_range(bad_port):
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError):
        DeployConfig(name="my-app", port=bad_port)


def test_deploy_config_port_in_range():
    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="my-app", port=8000)
    assert cfg.port == 8000


def test_deploy_config_gpus_negative():
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError):
        DeployConfig(name="my-app", port=8000, gpus=-1)


@pytest.mark.parametrize("good", ["512m", "2g", "1073741824", "1.5g", "512M", "2G", "512k", "10b"])
def test_deploy_config_memory_limit_valid(good):
    """P13 WP5: docker mem_limit grammar (bare bytes or b/k/m/g, case-insensitive)."""
    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="my-app", memory_limit=good)
    assert cfg.memory_limit == good


@pytest.mark.parametrize("bad", ["lots", "-1g", "512mb", "g", "", "1gb", "512 m"])
def test_deploy_config_memory_limit_invalid(bad):
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError, match="memory_limit"):
        DeployConfig(name="my-app", memory_limit=bad)


def test_deploy_config_memory_limit_optional():
    from nerdit.config.project import DeployConfig

    assert DeployConfig(name="my-app").memory_limit is None


@pytest.mark.parametrize("good", [0.5, 1, 1.5, 2.0, 8])
def test_deploy_config_cpu_limit_valid(good):
    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="my-app", cpu_limit=good)
    assert cfg.cpu_limit == good


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_deploy_config_cpu_limit_invalid(bad):
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError):
        DeployConfig(name="my-app", cpu_limit=bad)


def test_deploy_config_cpu_limit_optional():
    from nerdit.config.project import DeployConfig

    assert DeployConfig(name="my-app").cpu_limit is None


def test_deploy_config_gpus_zero_accepted():
    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="my-app", port=8000, gpus=0)
    assert cfg.gpus == 0


def test_deploy_config_port_optional():
    """port may be omitted (the buildpack picks a default); still name-validated."""
    from nerdit.config.project import DeployConfig

    cfg = DeployConfig(name="my-app")
    assert cfg.port is None
    assert cfg.gpus == 0


def test_deploy_config_port_bounds_enforced_when_set():
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError):
        DeployConfig(name="my-app", port=70000)


def test_load_project_config_deploy_without_port(tmp_path):
    from nerdit.config.project import load_project_config

    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[deploy]\nname = 'my-app'\nstart = 'npm start'\n")
    config = load_project_config(path=toml_file)
    assert config.deploy is not None
    assert config.deploy.port is None
    assert config.deploy.start == "npm start"


def test_load_project_config_deploy_invalid_name(tmp_path):
    from pydantic import ValidationError

    from nerdit.config.project import load_project_config

    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[deploy]\nname = 'My_App'\nport = 8000\n")
    with pytest.raises(ValidationError, match="DNS label"):
        load_project_config(path=toml_file)


# --- P20: [deploy].release ----------------------------------------------------
#
# ``release`` is the one deploy command that is VALIDATED (``start`` is
# deliberately left freeform): it GATES a deploy — a nonzero exit reverts the
# image — and it is shell-wrapped as ``/bin/sh -c``, so an embedded newline
# would hide a second statement from anyone reviewing the TOML.


def test_deploy_config_release_valid():
    from nerdit.config.project import DeployConfig

    cmd = "python manage.py migrate && npm run seed"
    assert DeployConfig(name="my-app", release=cmd).release == cmd


def test_deploy_config_release_optional():
    from nerdit.config.project import DeployConfig

    assert DeployConfig(name="my-app").release is None


@pytest.mark.parametrize(
    "bad",
    ["", " ", "\t", "   \t  "],
    ids=["empty", "space", "tab-only", "whitespace"],
)
def test_deploy_config_release_rejects_blank(bad):
    """Drop the key entirely to run no release step; a blank string would
    otherwise arm the crash marker for a command that does nothing."""
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError, match="non-empty command"):
        DeployConfig(name="my-app", release=bad)


@pytest.mark.parametrize(
    "bad",
    [
        "migrate\nrm -rf /",  # LF: a hidden second statement
        "migrate\r\nrm -rf /",  # CRLF
        "migrate\ttrailing",  # TAB
        "migrate\x00hidden",  # NUL
        "migrate\x7f",  # DEL
        "migrate\x1b[2Kforged",  # ESC — terminal escape / log forging
        "migrate\x85next-line",  # C1 NEL
    ],
    ids=["lf", "crlf", "tab", "nul", "del", "esc", "c1-nel"],
)
def test_deploy_config_release_rejects_control_characters(bad):
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    with pytest.raises(ValidationError, match="control characters"):
        DeployConfig(name="my-app", release=bad)


def test_deploy_config_release_validator_message_never_echoes_the_value():
    """Security: the control characters are exactly what makes the value unsafe
    to interpolate into a message that then rides the 422 envelope, ``job_logs``,
    ``/diagnose`` and the archive-first audit trail.

    Asserted on ``errors()[0]["msg"]`` because that — not ``str(exc)`` — is what
    both 422 builders interpolate (``validate_deploy_fields`` in
    ``config/app_config.py`` and ``_parse_deploy_defaults`` in
    ``daemon/deploy_pipeline.py``). Pydantic's own ``str(ValidationError)``
    repr does carry an ``input=`` echo; it must never reach a response, which
    the route-tier twin in ``tests/test_routes_app_config.py`` pins end to end.
    """
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    payload = "migrate\x1b[2K\x00FORGED-AUDIT-LINE"
    with pytest.raises(ValidationError) as exc:
        DeployConfig(name="my-app", release=payload)
    msg = exc.value.errors()[0]["msg"]
    assert "FORGED-AUDIT-LINE" not in msg
    assert "\x00" not in msg and "\x1b" not in msg
    assert "control characters" in msg


def test_deploy_config_release_max_length():
    from pydantic import ValidationError

    from nerdit.config.project import DeployConfig

    assert DeployConfig(name="my-app", release="x" * 4096).release == "x" * 4096
    with pytest.raises(ValidationError):
        DeployConfig(name="my-app", release="x" * 4097)


def test_load_project_config_parses_release(tmp_path):
    from nerdit.config.project import load_project_config

    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text(
        "[deploy]\nname = 'my-app'\nrelease = 'alembic upgrade head && echo done'\n"
    )
    config = load_project_config(path=toml_file)
    assert config.deploy.release == "alembic upgrade head && echo done"


def test_load_project_config_rejects_multiline_release(tmp_path):
    """A TOML triple-quoted string is the natural way to write a multi-line
    migration; it must be refused at parse time, not silently shell-wrapped."""
    from pydantic import ValidationError

    from nerdit.config.project import load_project_config

    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text('[deploy]\nname = "my-app"\nrelease = """\nmigrate\nseed\n"""\n')
    with pytest.raises(ValidationError, match="control characters"):
        load_project_config(path=toml_file)
