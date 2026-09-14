"""Exercise Node detection, runtime selection and Dockerfile assembly together."""

import json

import pytest

from nerdit.config.project import DeployConfig
from nerdit.core.builder import BuildpackNotSupported, detect


def _manifest(tmp_path, **fields):
    pkg = {"name": "fixture", "version": "1.0.0", **fields}
    (tmp_path / "package.json").write_text(json.dumps(pkg))


@pytest.mark.parametrize("group", ["dependencies", "devDependencies"])
def test_next_uses_declared_runtime_manager_and_port(tmp_path, group):
    _manifest(
        tmp_path,
        **{group: {"next": "16.0.0"}},
        engines={"node": "^22"},
        packageManager="pnpm@10.34.5",
        scripts={"build": "next build", "start": "next start"},
    )
    plan = detect(tmp_path, DeployConfig(name="fixture", port=8080))
    text = plan.dockerfile_text
    assert text is not None
    assert (plan.framework, plan.node_version, plan.package_manager) == (
        "nextjs",
        "22.23.2",
        "pnpm@10.34.5",
    )
    assert plan.port == 8080 and plan.start_command == "pnpm start"
    assert "FROM node:22.23.2-slim" in text and "EXPOSE 8080" in text
    assert text.index("RUN pnpm install") < text.index("RUN pnpm run build")
    assert text.index("COPY . .") < text.index("RUN pnpm run build")
    assert text.index("RUN pnpm run build") < text.index('CMD ["pnpm", "start"]')


@pytest.mark.parametrize(
    "scripts",
    [{}, {"build": "next build"}, {"start": "next start"}, {"build": "  ", "start": "next start"}],
)
def test_next_missing_server_scripts_fails_before_container_build(tmp_path, scripts):
    _manifest(tmp_path, dependencies={"next": "16.0.0"}, scripts=scripts)
    with pytest.raises(BuildpackNotSupported, match="Next.js.*build and start.*Dockerfile"):
        detect(tmp_path)


def test_next_explicit_start_keeps_build_step(tmp_path):
    _manifest(tmp_path, dependencies={"next": "16.0.0"}, scripts={"build": "next build"})
    command = "node .next/standalone/server.js"
    plan = detect(tmp_path, DeployConfig(name="fixture", start=command))
    assert plan.start_command == command
    assert plan.dockerfile_text is not None
    assert "RUN npm run build" in plan.dockerfile_text
    assert f'CMD ["sh", "-c", "{command}"]' in plan.dockerfile_text


def test_custom_dockerfile_bypasses_invalid_node_metadata(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "package.json").write_bytes(b"\xff SECRET")
    (tmp_path / ".nvmrc").write_text("SECRET")
    (tmp_path / "pnpm-lock.yaml").write_text("invalid")
    (tmp_path / "yarn.lock").write_text("invalid")
    plan = detect(tmp_path)
    assert plan.language == "dockerfile" and plan.dockerfile_text is None
    assert plan.framework is None and plan.package_manager is None and plan.node_version is None


@pytest.mark.parametrize(
    ("manager", "lockfile", "lock", "install"),
    [
        (
            "pnpm@10.34.5",
            "pnpm-lock.yaml",
            "lockfileVersion: '9.0'\n",
            "pnpm install --frozen-lockfile --prod=false",
        ),
        (
            "yarn@1.22.22",
            "yarn.lock",
            "# yarn lockfile v1\n",
            "yarn install --frozen-lockfile --production=false",
        ),
        ("yarn@4.18.0", "yarn.lock", "__metadata:\n  version: 8\n", "yarn install --immutable"),
    ],
)
def test_simple_locked_managers_preserve_cache_build_and_start(
    tmp_path, manager, lockfile, lock, install
):
    _manifest(
        tmp_path, packageManager=manager, scripts={"build": "tsc", "start": "node dist/main.js"}
    )
    (tmp_path / lockfile).write_text(lock)
    plan = detect(tmp_path)
    text = plan.dockerfile_text
    assert text is not None
    name = manager.split("@")[0]
    assert plan.framework == "node" and plan.package_manager == manager
    assert text.index(f"COPY package.json {lockfile} ./") < text.index(f"RUN {install}")
    assert (
        text.index(f"RUN {install}") < text.index("COPY . .") < text.index(f"RUN {name} run build")
    )
    assert f'CMD ["{name}", "start"]' in text
    assert "NODE_ENV=production" not in text and "--omit=dev" not in text


@pytest.mark.parametrize(
    ("fields", "config"),
    [
        ({"scripts": {"prepare": "node scripts/prepare.js"}}, None),
        ({"scripts": {"postinstall": "node scripts/setup.js"}}, None),
        ({"dependencies": {"local-package": "file:./packages/local"}}, None),
        ({"devDependencies": {"local-package": "link:./packages/local"}}, None),
        ({}, ".npmrc"),
        ({}, ".yarnrc.yml"),
        ({}, "pnpm-workspace.yaml"),
    ],
)
def test_source_dependent_installs_copy_source_first(tmp_path, fields, config):
    _manifest(tmp_path, **fields)
    if config:
        (tmp_path / config).write_text("# fixture config\n")
    text = detect(tmp_path).dockerfile_text
    assert text is not None
    assert text.index("COPY . .") < text.index("RUN npm install --include=dev")
    assert "COPY package*.json ./" not in text


def test_yarn_pnp_fallback_loads_dependency_map(tmp_path):
    _manifest(tmp_path, packageManager="yarn@4.18.0")
    text = detect(tmp_path).dockerfile_text
    assert text is not None and 'CMD ["yarn", "node", "server.js"]' in text


@pytest.mark.parametrize(
    "raw",
    [
        b"\xffSECRET",
        b"{SECRET",
        b'["SECRET"]',
        b'"SECRET"',
        b"null",
        b"123",
        b'{"scripts": "SECRET"}',
        b'{"scripts": {"build": ["SECRET"]}}',
        b'{"scripts": {"start": {"SECRET": true}}}',
        b'{"scripts": {"build": null}}',
        b'{"dependencies": "SECRET"}',
        b'{"devDependencies": ["SECRET"]}',
    ],
)
def test_invalid_manifests_fail_with_safe_buildpack_error(tmp_path, raw):
    (tmp_path / "package.json").write_bytes(raw)
    with pytest.raises(BuildpackNotSupported) as error:
        detect(tmp_path)
    assert "SECRET" not in str(error.value)
    assert str(tmp_path) not in str(error.value)


def test_corepack_cannot_replace_resolved_pin_from_secondary_metadata(tmp_path):
    _manifest(
        tmp_path,
        devEngines={
            "packageManager": {"name": "yarn", "version": "https://example.invalid/tool.js"}
        },
    )
    plan = detect(tmp_path)
    assert plan.package_manager == "npm@11.19.1"
    assert "COREPACK_ENABLE_PROJECT_SPEC=0" in plan.dockerfile_text
    assert "COREPACK_DEFAULT_TO_LATEST=0" in plan.dockerfile_text
    assert "example.invalid" not in plan.dockerfile_text


@pytest.mark.parametrize("next_app", [False, True])
def test_yarn4_next_uses_turbopack_compatible_layout(tmp_path, next_app):
    _manifest(
        tmp_path,
        packageManager="yarn@4.18.0",
        dependencies={"next": "16.3.4"} if next_app else {},
        scripts={"build": "next build", "start": "next start"},
    )
    plan = detect(tmp_path)
    assert ("ENV YARN_NODE_LINKER=node-modules" in plan.dockerfile_text) is next_app
    assert "RUN yarn run build" in plan.dockerfile_text


def test_next_public_env_args_declared_after_install_before_build(tmp_path):
    _manifest(
        tmp_path,
        packageManager="pnpm@10.34.5",
        dependencies={"next": "16.0.0"},
        scripts={"build": "next build", "start": "next start"},
    )
    text = detect(
        tmp_path,
        DeployConfig(
            name="fixture",
            port=3000,
            build_settings={"public_env": {"NEXT_PUBLIC_URL": "https://x.test"}},
        ),
    ).dockerfile_text
    assert text.index("RUN pnpm install") < text.index("ARG NEXT_PUBLIC_URL")
    assert text.index("ARG NEXT_PUBLIC_URL") < text.index("RUN pnpm run build")
    assert "NEXT_PUBLIC_URL=" not in text  # values travel as --build-arg, never as a default
