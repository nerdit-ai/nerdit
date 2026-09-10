"""Package-manager declarations are bounded container inputs, never host commands."""

import json

import pytest

from nerdit.core.node_packages import node_install_needs_source, resolve_node_packages


@pytest.mark.parametrize(
    ("pin", "command"),
    [
        ("npm@10.9.3", "npm install --include=dev"),
        ("npm@11.19.1", "npm install --include=dev"),
        ("pnpm@9.15.9", "pnpm install --no-frozen-lockfile --prod=false"),
        ("pnpm@10.34.5", "pnpm install --no-frozen-lockfile --prod=false"),
        ("pnpm@11.0.0", "pnpm install --no-frozen-lockfile --prod=false"),
        ("pnpm@12.0.0", "pnpm install --no-frozen-lockfile --prod=false"),
        ("yarn@1.22.22", "yarn install --production=false"),
        ("yarn@4.18.0", "yarn install --no-immutable"),
    ],
)
def test_exact_supported_pin(tmp_path, pin, command):
    plan = resolve_node_packages(tmp_path, {"packageManager": pin})
    assert f"{plan.name}@{plan.version}" == pin
    assert plan.install == command
    assert plan.lockfile is None


def test_corepack_integrity_pin_preserved(tmp_path):
    version = "10.34.5+sha512." + "a" * 128
    plan = resolve_node_packages(tmp_path, {"packageManager": f"pnpm@{version}"})
    assert plan.version == version
    assert plan.name == "pnpm"


@pytest.mark.parametrize(
    "declaration",
    [
        None,
        42,
        {},
        [],
        "",
        "bun@1.2.0",
        "npm@9.0.0",
        "pnpm@8.0.0",
        "yarn@2.0.0",
        "yarn@3.0.0",
        "npm@latest",
        "npm@^11.0.0",
        "npm@11",
        "npm@11.0.0-beta.1",
        "npm@011.0.0",
        "npm@11.0.0\nRUN echo sensitive-marker",
        "npm@https://sensitive-marker.invalid/tool.tgz",
        "npm@11.0.0;echo sensitive-marker",
        "$(sensitive-marker)@11.0.0",
        "yarn@4.18.0+sha512.nothex",
        "x" * 257,
    ],
)
def test_bad_declaration_rejected_without_echo(tmp_path, declaration):
    with pytest.raises(ValueError) as error:
        resolve_node_packages(tmp_path, {"packageManager": declaration})
    assert "sensitive-marker" not in str(error.value)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize(
    ("filename", "content", "manager", "version", "command"),
    [
        ("package-lock.json", '{"lockfileVersion":3}', "npm", "11.19.1", "npm ci --include=dev"),
        ("npm-shrinkwrap.json", '{"lockfileVersion":2}', "npm", "11.19.1", "npm ci --include=dev"),
        (
            "pnpm-lock.yaml",
            "lockfileVersion: '9.0'\n",
            "pnpm",
            "10.34.5",
            "pnpm install --frozen-lockfile --prod=false",
        ),
        (
            "yarn.lock",
            "# yarn lockfile v1\n",
            "yarn",
            "1.22.22",
            "yarn install --frozen-lockfile --production=false",
        ),
        (
            "yarn.lock",
            "__metadata:\n  version: 8\n  cacheKey: 10c0\n",
            "yarn",
            "4.9.4",
            "yarn install --immutable",
        ),
    ],
)
def test_lockfile_inference(tmp_path, filename, content, manager, version, command):
    (tmp_path / filename).write_text(content)
    plan = resolve_node_packages(tmp_path, {})
    assert (plan.name, plan.version, plan.lockfile, plan.install) == (
        manager,
        version,
        filename,
        command,
    )


@pytest.mark.parametrize("version", [1, 2, 3])
def test_supported_npm_lock_versions(tmp_path, version):
    (tmp_path / "package-lock.json").write_text(json.dumps({"lockfileVersion": version}))
    assert resolve_node_packages(tmp_path, {}).install == "npm ci --include=dev"


@pytest.mark.parametrize("pin", ["pnpm@9.15.9", "pnpm@10.34.5", "pnpm@11.0.0", "pnpm@12.0.0"])
def test_pnpm_supported_pins_accept_current_lock_version(tmp_path, pin):
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")
    assert "--frozen-lockfile" in resolve_node_packages(tmp_path, {"packageManager": pin}).install


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("package-lock.json", "not json"),
        ("package-lock.json", "[]"),
        ("package-lock.json", '{"lockfileVersion":true}'),
        ("package-lock.json", '{"lockfileVersion":"3"}'),
        ("npm-shrinkwrap.json", '{"lockfileVersion":99}'),
        ("pnpm-lock.yaml", "lockfileVersion: '6.0'\n"),
        ("pnpm-lock.yaml", "lockfileVersion: '9.0\"\n"),
        ("pnpm-lock.yaml", "lockfileVersion: \"9.0'\n"),
        ("pnpm-lock.yaml", "lockfileVersion: '9.0\n"),
        ("pnpm-lock.yaml", "nested:\n  lockfileVersion: '9.0'\n"),
        ("yarn.lock", "__metadata:\n  version: 6\n"),
        ("yarn.lock", "not a lockfile"),
    ],
)
def test_unknown_lock_format_rejected(tmp_path, filename, content):
    (tmp_path / filename).write_text(content)
    with pytest.raises(ValueError, match="lockfile"):
        resolve_node_packages(tmp_path, {})


@pytest.mark.parametrize(
    "filenames",
    [
        ("package-lock.json", "npm-shrinkwrap.json"),
        ("package-lock.json", "pnpm-lock.yaml"),
        ("yarn.lock", "bun.lock"),
        ("bun.lock", "bun.lockb"),
    ],
)
def test_multiple_lockfiles_rejected(tmp_path, filenames):
    for filename in filenames:
        (tmp_path / filename).write_text("")
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_node_packages(tmp_path, {})


@pytest.mark.parametrize("filename", ["bun.lock", "bun.lockb"])
def test_bun_requires_dockerfile(tmp_path, filename):
    (tmp_path / filename).write_text("")
    with pytest.raises(ValueError, match="Bun builds require a Dockerfile"):
        resolve_node_packages(tmp_path, {})


@pytest.mark.parametrize("pin", ["pnpm@10.34.5", "yarn@4.18.0"])
def test_declared_manager_must_match_lock(tmp_path, pin):
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}')
    with pytest.raises(ValueError, match="conflicts"):
        resolve_node_packages(tmp_path, {"packageManager": pin})


@pytest.mark.parametrize(
    ("pin", "content"),
    [
        ("yarn@4.18.0", "# yarn lockfile v1\n"),
        ("yarn@1.22.22", "__metadata:\n  version: 8\n"),
    ],
)
def test_yarn_major_must_match_lock_format(tmp_path, pin, content):
    (tmp_path / "yarn.lock").write_text(content)
    with pytest.raises(ValueError, match="Yarn version conflicts"):
        resolve_node_packages(tmp_path, {"packageManager": pin})


def test_modern_yarn_lock_inference(tmp_path):
    (tmp_path / "yarn.lock").write_text("__metadata:\n  version: 10\n  cacheKey: 10c0\n")
    plan = resolve_node_packages(tmp_path, {})
    assert (plan.name, plan.version, plan.install) == ("yarn", "4.18.0", "yarn install --immutable")


def test_dangling_lockfile_symlink_rejected(tmp_path):
    (tmp_path / "pnpm-lock.yaml").symlink_to(tmp_path / "missing-lock.yaml")
    with pytest.raises(ValueError, match="regular file"):
        resolve_node_packages(tmp_path, {})


def test_lockfile_symlink_rejected(tmp_path):
    target = tmp_path / "elsewhere.json"
    target.write_text('{"lockfileVersion":3}')
    (tmp_path / "package-lock.json").symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        resolve_node_packages(tmp_path, {})


@pytest.mark.parametrize("name", ["preinstall", "install", "postinstall", "prepare"])
def test_install_lifecycle_requires_source(tmp_path, name):
    assert node_install_needs_source(tmp_path, {"scripts": {name: "node scripts/setup.js"}})


@pytest.mark.parametrize(
    "filename", [".npmrc", ".yarnrc", ".yarnrc.yml", ".yarn", "pnpm-workspace.yaml"]
)
def test_manager_config_requires_source(tmp_path, filename):
    (tmp_path / filename).touch()
    assert node_install_needs_source(tmp_path, {})


@pytest.mark.parametrize("key", ["workspaces", "pnpm"])
def test_workspace_or_pnpm_settings_require_source(tmp_path, key):
    assert node_install_needs_source(tmp_path, {key: {}})


@pytest.mark.parametrize(
    "group", ["dependencies", "devDependencies", "optionalDependencies", "resolutions"]
)
@pytest.mark.parametrize(
    "reference", ["file:packages/lib", "link:packages/lib", "workspace:*", "./packages/lib"]
)
def test_local_dependency_requires_source(tmp_path, group, reference):
    assert node_install_needs_source(tmp_path, {group: {"local": reference}})


def test_ordinary_build_keeps_manifest_install_cache(tmp_path):
    assert not node_install_needs_source(
        tmp_path,
        {
            "scripts": {"build": "next build", "start": "next start"},
            "dependencies": {"next": "16.1.0"},
        },
    )


@pytest.mark.parametrize("script", ["prepublish", "preprepare", "postprepare"])
def test_additional_npm_lifecycle_requires_source(tmp_path, script):
    assert node_install_needs_source(tmp_path, {"scripts": {script: "node scripts/setup.js"}})


def test_implicit_native_build_requires_source(tmp_path):
    (tmp_path / "binding.gyp").write_text("{}")
    assert node_install_needs_source(tmp_path, {})
