"""Unit tests for the P4 builder (pure detection + Dockerfile generation)."""

import json

import pytest

from nerdit.config.defaults import (
    APP_IMAGE_REPO_PREFIX,
    app_image_repo,
    app_image_tag,
)
from nerdit.config.project import DeployConfig
from nerdit.core.builder import (
    DEFAULT_NODE_PORT,
    DEFAULT_PYTHON_PORT,
    GENERATED_DOCKERFILE_NAME,
    NODE_BASE_IMAGE,
    PYTHON_BASE_IMAGE,
    BuildpackNotSupported,
    detect,
)


def _write_package_json(tmp_path, scripts=None, workspaces=None):
    pkg = {"name": "fixture-app", "version": "1.0.0"}
    if scripts is not None:
        pkg["scripts"] = scripts
    if workspaces is not None:
        pkg["workspaces"] = workspaces
    (tmp_path / "package.json").write_text(json.dumps(pkg))


def _write_requirements(tmp_path, deps=None):
    (tmp_path / "requirements.txt").write_text("\n".join(deps or ["fastapi"]) + "\n")


def _write_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0.1.0'\n")


def _deploy_cfg(**overrides):
    base = {"name": "my-app", "port": 8000}
    base.update(overrides)
    return DeployConfig(**base)


def _idx(text, needle):
    """Index of *needle* in *text*; fails the test if absent."""
    pos = text.find(needle)
    assert pos != -1, f"{needle!r} not found in generated Dockerfile"
    return pos


# --- Detection precedence ---


def test_dockerfile_wins_over_package_json(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM alpine\n")
    _write_package_json(tmp_path, scripts={"build": "next build", "start": "next start"})

    plan = detect(tmp_path)

    assert plan.language == "dockerfile"
    assert plan.dockerfile_name == "Dockerfile"
    assert plan.dockerfile_text is None  # passthrough: use the on-disk file


def test_dockerfile_passthrough_carries_deploy_hints(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM alpine\n")

    plan = detect(tmp_path, deploy_cfg=_deploy_cfg(start="./run.sh"))

    assert plan.port == 8000
    assert plan.start_command == "./run.sh"


def test_dockerfile_passthrough_without_deploy_cfg(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM alpine\n")

    plan = detect(tmp_path)

    assert plan.port is None
    assert plan.start_command is None


def test_python_wins_over_node_markers(tmp_path):
    """A mixed repo (requirements.txt + package.json) builds as Python."""
    _write_requirements(tmp_path)
    _write_package_json(tmp_path)

    plan = detect(tmp_path)

    assert plan.language == "python"  # precedence preserved: Python before Node
    assert plan.dockerfile_name == GENERATED_DOCKERFILE_NAME


# --- Node buildpack ---


def test_node_generates_dockerfile(tmp_path):
    _write_package_json(tmp_path, scripts={"start": "node index.js"})

    plan = detect(tmp_path)

    assert plan.language == "node"
    assert plan.dockerfile_name == GENERATED_DOCKERFILE_NAME
    assert plan.dockerfile_text
    assert NODE_BASE_IMAGE in plan.dockerfile_text
    assert 'CMD ["npm", "start"]' in plan.dockerfile_text
    assert plan.start_command == "npm start"
    assert plan.port == DEFAULT_NODE_PORT
    assert f"EXPOSE {DEFAULT_NODE_PORT}" in plan.dockerfile_text


def test_node_layer_order_deps_copy_then_install_then_source(tmp_path):
    """Node cache layering (P13 WP8): COPY package*.json → install → COPY . .

    The dependency-install layer must precede the full-source copy so a
    source-only edit reuses the cached install layer.
    """
    _write_package_json(tmp_path, scripts={"start": "node index.js"})

    text = detect(tmp_path).dockerfile_text

    deps_copy = _idx(text, "COPY package*.json ./")
    install = _idx(text, "RUN npm install")
    source_copy = _idx(text, "COPY . .")
    assert deps_copy < install < source_copy


@pytest.mark.parametrize("workspaces", [None, ["packages/*"]])
@pytest.mark.parametrize("locked", [False, True])
def test_node_build_script_runs_after_source_and_install(tmp_path, workspaces, locked):
    _write_package_json(
        tmp_path, scripts={"build": "next build", "start": "next start"}, workspaces=workspaces
    )
    if locked:
        (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}')

    plan = detect(tmp_path)
    text = plan.dockerfile_text

    build = _idx(text, "RUN npm run build")
    assert _idx(text, "COPY . .") < build
    assert _idx(text, "RUN npm ci" if locked else "RUN npm install") < build
    assert build < _idx(text, 'CMD ["npm", "start"]')
    assert plan.start_command == "npm start"


@pytest.mark.parametrize(
    "scripts",
    [None, {}, {"start": "node index.js"}, {"build": ""}, {"build": "  "}],
)
def test_node_without_valid_build_script_skips_build(tmp_path, scripts):
    _write_package_json(tmp_path, scripts=scripts)

    assert "RUN npm run build" not in detect(tmp_path).dockerfile_text


def test_node_npm_ci_with_lockfile(tmp_path):
    _write_package_json(tmp_path, scripts={"start": "node index.js"})
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 3}')

    plan = detect(tmp_path)

    assert "RUN npm ci" in plan.dockerfile_text
    assert "RUN npm install" not in plan.dockerfile_text


def test_node_npm_install_without_lockfile(tmp_path):
    _write_package_json(tmp_path, scripts={"start": "node index.js"})

    plan = detect(tmp_path)

    assert "RUN npm install" in plan.dockerfile_text
    assert "RUN npm ci" not in plan.dockerfile_text


def test_node_fallback_cmd_without_start_script(tmp_path):
    _write_package_json(tmp_path)  # no scripts at all

    plan = detect(tmp_path)

    assert 'CMD ["node", "server.js"]' in plan.dockerfile_text
    assert plan.start_command is None


def test_node_deploy_start_overrides_scripts_start(tmp_path):
    _write_package_json(tmp_path, scripts={"start": "node index.js"})

    plan = detect(tmp_path, deploy_cfg=_deploy_cfg(start="node custom.js --prod"))

    assert plan.start_command == "node custom.js --prod"
    assert 'CMD ["sh", "-c", "node custom.js --prod"]' in plan.dockerfile_text
    assert 'CMD ["npm", "start"]' not in plan.dockerfile_text


def test_node_deploy_port_overrides_default(tmp_path):
    _write_package_json(tmp_path, scripts={"start": "node index.js"})

    plan = detect(tmp_path, deploy_cfg=_deploy_cfg(port=8080))

    assert plan.port == 8080
    assert "EXPOSE 8080" in plan.dockerfile_text


def test_node_unset_port_falls_back_to_default(tmp_path):
    """A DeployConfig with no port still lets the Node buildpack pick its default."""
    _write_package_json(tmp_path, scripts={"start": "node index.js"})

    plan = detect(tmp_path, deploy_cfg=DeployConfig(name="my-app"))  # port omitted

    assert plan.port == DEFAULT_NODE_PORT
    assert f"EXPOSE {DEFAULT_NODE_PORT}" in plan.dockerfile_text


def test_dockerfile_passthrough_unset_port_is_none(tmp_path):
    """Passthrough with an unset [deploy].port carries port=None (no fabrication)."""
    (tmp_path / "Dockerfile").write_text("FROM alpine\n")

    plan = detect(tmp_path, deploy_cfg=DeployConfig(name="my-app"))

    assert plan.port is None


def test_node_invalid_package_json_raises(tmp_path):
    (tmp_path / "package.json").write_text("{not json")

    with pytest.raises(BuildpackNotSupported, match="not valid JSON"):
        detect(tmp_path)


# --- F3-BUILDER: fallback ordering for workspaces / tree-referencing reqs ---


def test_f3_builder_node_workspaces_falls_back_to_copy_first(tmp_path):
    """A root package.json with "workspaces" must copy the whole tree BEFORE
    install (npm needs packages/*/package.json), and drop the manifests glob."""
    _write_package_json(tmp_path, scripts={"start": "node index.js"}, workspaces=["packages/*"])

    text = detect(tmp_path).dockerfile_text

    source_copy = _idx(text, "COPY . .")
    install = _idx(text, "RUN npm install")
    assert source_copy < install, "workspaces build must copy source before install"
    assert "COPY package*.json ./" not in text


def test_f3_builder_plain_manifests_keep_fast_path(tmp_path):
    """No workspaces key → the manifests-first cache optimization is preserved."""
    _write_package_json(tmp_path, scripts={"start": "node index.js"})

    text = detect(tmp_path).dockerfile_text

    deps_copy = _idx(text, "COPY package*.json ./")
    install = _idx(text, "RUN npm install")
    source_copy = _idx(text, "COPY . .")
    assert deps_copy < install < source_copy


@pytest.mark.parametrize(
    "dep",
    ["-e .", ".", "./libs/foo", "../shared", "-r other.txt", "-c constraints.txt", "file:./pkg"],
)
def test_f3_builder_requirements_tree_refs_fall_back(tmp_path, dep):
    """A tree-referencing requirements.txt entry forces install-after-copy."""
    _write_requirements(tmp_path, deps=["fastapi", dep])

    text = detect(tmp_path).dockerfile_text

    source_copy = _idx(text, "COPY . .")
    install = _idx(text, "RUN pip install --no-cache-dir -r requirements.txt")
    assert source_copy < install, f"{dep!r} must install after COPY . ."
    assert "COPY requirements.txt ./" not in text


def test_f3_builder_plain_requirements_keep_fast_path(tmp_path):
    """Plain pinned requirements → the pin-file-first cache optimization stays."""
    _write_requirements(tmp_path, deps=["fastapi==0.110", "uvicorn>=0.30"])

    text = detect(tmp_path).dockerfile_text

    deps_copy = _idx(text, "COPY requirements.txt ./")
    install = _idx(text, "RUN pip install --no-cache-dir -r requirements.txt")
    source_copy = _idx(text, "COPY . .")
    assert deps_copy < install < source_copy


# --- Python buildpack ---


def test_python_requirements_generates_dockerfile(tmp_path):
    _write_requirements(tmp_path)

    plan = detect(tmp_path)

    assert plan.language == "python"
    assert plan.dockerfile_name == GENERATED_DOCKERFILE_NAME
    assert plan.dockerfile_text
    assert PYTHON_BASE_IMAGE in plan.dockerfile_text
    assert "python:3.11-slim" in plan.dockerfile_text
    assert "RUN pip install --no-cache-dir -r requirements.txt" in plan.dockerfile_text
    assert "EXPOSE 8000" in plan.dockerfile_text
    assert 'CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]' in plan.dockerfile_text
    assert plan.start_command is None


def test_python_requirements_layer_order_deps_first(tmp_path):
    """Python requirements path (P13 WP8): COPY requirements.txt → install → COPY . ."""
    _write_requirements(tmp_path)

    text = detect(tmp_path).dockerfile_text

    deps_copy = _idx(text, "COPY requirements.txt ./")
    install = _idx(text, "RUN pip install --no-cache-dir -r requirements.txt")
    source_copy = _idx(text, "COPY . .")
    assert deps_copy < install < source_copy


def test_python_pyproject_uses_pep517_install(tmp_path):
    _write_pyproject(tmp_path)

    plan = detect(tmp_path)

    assert plan.language == "python"
    assert "RUN pip install --no-cache-dir ." in plan.dockerfile_text
    assert "-r requirements.txt" not in plan.dockerfile_text


def test_python_pyproject_install_stays_after_source_copy(tmp_path):
    """The pyproject-only path MUST NOT reorder: PEP 517 needs the full tree,
    so ``pip install .`` stays after ``COPY . .`` (the must-not-reorder case)."""
    _write_pyproject(tmp_path)

    text = detect(tmp_path).dockerfile_text

    source_copy = _idx(text, "COPY . .")
    install = _idx(text, "RUN pip install --no-cache-dir .")
    assert source_copy < install


def test_python_deploy_start_overrides_uvicorn_fallback(tmp_path):
    _write_requirements(tmp_path)

    plan = detect(tmp_path, deploy_cfg=_deploy_cfg(start="gunicorn app:app"))

    assert plan.start_command == "gunicorn app:app"
    assert 'CMD ["sh", "-c", "gunicorn app:app"]' in plan.dockerfile_text
    # The uvicorn fallback CMD is not emitted when [deploy].start is set.
    assert "uvicorn main:app --host" not in plan.dockerfile_text


def test_python_deploy_port_overrides_default(tmp_path):
    _write_requirements(tmp_path)

    plan = detect(tmp_path, deploy_cfg=_deploy_cfg(port=8080))

    assert plan.port == 8080
    assert "EXPOSE 8080" in plan.dockerfile_text
    assert 'CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8080"]' in plan.dockerfile_text


def test_python_unset_port_falls_back_to_default(tmp_path):
    """A DeployConfig with no port lets the Python buildpack pick its default."""
    _write_requirements(tmp_path)

    plan = detect(tmp_path, deploy_cfg=DeployConfig(name="my-app"))  # port omitted

    assert plan.port == DEFAULT_PYTHON_PORT
    assert f"EXPOSE {DEFAULT_PYTHON_PORT}" in plan.dockerfile_text
    assert 'CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]' in plan.dockerfile_text


def test_dockerfile_wins_over_python_markers(tmp_path):
    """An existing Dockerfile beats requirements.txt (passthrough)."""
    (tmp_path / "Dockerfile").write_text("FROM alpine\n")
    _write_requirements(tmp_path)

    plan = detect(tmp_path)

    assert plan.language == "dockerfile"
    assert plan.dockerfile_name == "Dockerfile"
    assert plan.dockerfile_text is None


def test_python_detect_accepts_str_path(tmp_path):
    _write_requirements(tmp_path)

    plan = detect(str(tmp_path))

    assert plan.language == "python"


# --- Unsupported / empty folders ---


def test_empty_folder_raises(tmp_path):
    with pytest.raises(BuildpackNotSupported, match="no Dockerfile, package.json"):
        detect(tmp_path)


def test_no_buildpack_message_is_path_free(tmp_path):
    """The message reaches remote callers verbatim as ``deploy.no_buildpack``;
    the extraction dir is daemon-internal and must never ride along (P29
    security-review note)."""
    with pytest.raises(BuildpackNotSupported) as exc:
        detect(tmp_path)
    assert str(tmp_path) not in str(exc.value)


def test_detect_accepts_str_path(tmp_path):
    _write_package_json(tmp_path, scripts={"start": "node index.js"})

    plan = detect(str(tmp_path))

    assert plan.language == "node"


# --- defaults.py image-tag helpers ---


def test_app_image_helpers():
    assert APP_IMAGE_REPO_PREFIX == "nerdit-app"
    assert app_image_repo("my-app") == "nerdit-app/my-app"
    assert app_image_tag("my-app", 3) == "nerdit-app/my-app:3"


def test_preset_selects_node_in_mixed_root(tmp_path):
    _write_package_json(tmp_path, scripts={"start": "node app.js"})
    _write_requirements(tmp_path)
    assert detect(str(tmp_path)).language == "python"
    assert (
        detect(str(tmp_path), _deploy_cfg(build_settings={"preset": "python"})).language == "python"
    )
    plan = detect(str(tmp_path), _deploy_cfg(build_settings={"preset": "node"}))
    assert plan.language == "node"
    assert plan.start_command == "npm start"


@pytest.mark.parametrize("preset", ["node", "nextjs", "python", "dockerfile"])
def test_preset_requires_its_project_files(tmp_path, preset):
    with pytest.raises(BuildpackNotSupported, match="requires"):
        detect(str(tmp_path), _deploy_cfg(build_settings={"preset": preset}))


def test_next_preset_requires_dependency_and_cannot_bypass_next_validation(tmp_path):
    _write_package_json(tmp_path, scripts={"start": "node server.js", "build": "echo build"})
    with pytest.raises(BuildpackNotSupported, match="declared next dependency"):
        detect(str(tmp_path), _deploy_cfg(build_settings={"preset": "nextjs"}))
    (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"next": "16.3.4"}}))
    with pytest.raises(BuildpackNotSupported, match="require build and start"):
        detect(str(tmp_path), _deploy_cfg(build_settings={"preset": "node"}))
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {"next": "16.3.4"},
                "scripts": {"build": "next build", "start": "next start"},
            }
        )
    )
    assert (
        detect(str(tmp_path), _deploy_cfg(build_settings={"preset": "nextjs"})).framework
        == "nextjs"
    )


@pytest.mark.parametrize("preset", ["node", "nextjs", "python", "dockerfile"])
def test_dockerfile_remains_authoritative_with_preset(tmp_path, preset):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "package.json").write_text("malformed")
    plan = detect(str(tmp_path), _deploy_cfg(build_settings={"preset": preset}))
    assert plan.language == "dockerfile"
    assert bool(plan.warnings) == (preset != "dockerfile")
