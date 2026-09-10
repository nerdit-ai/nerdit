"""Override precedence and validation for the shared generated-image planner."""

import json

import pytest
from pydantic import ValidationError

from nerdit.config.build import BuildSettings
from nerdit.config.project import DeployConfig, load_project_config
from nerdit.core.builder import BuildpackNotSupported, detect


@pytest.mark.parametrize(
    "settings",
    [
        {"install": ""},
        {"build": True},
        {"build": 0},
        {"build": 1},
        {"build": "echo ok\nFROM bad"},
        {"start": "x\x00y"},
        {"install": "a" * 4097},
        {"install": "echo ${secrets.TOKEN}"},
        {"build": "echo ${github.token}"},
        {"subdir": "../app"},
        {"subdir": "/app"},
        {"subdir": "app/../app"},
        {"subdir": "app\\child"},
        {"subdir": "C:/app"},
        {"subdir": "app\nchild"},
        {"subdir": "a" * 513},
        {"node_version": "24"},
        {"node_version": 24},
        {"package_manager": "npm@latest"},
        {"package_manager": "npm@https://example.com"},
        {"package_manager": "pnpm@8.0.0"},
        {"env": {"PUBLIC_KEY": "x"}},
        {"build_env": {"TOKEN": "x"}},
        {"unknown": "x"},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValidationError):
        BuildSettings.model_validate(settings)


def test_rejected_commands_are_not_echoed():
    sentinel = "never-print-this\n"
    with pytest.raises(ValidationError) as exc:
        DeployConfig(name="app", build_settings={"install": sentinel})
    assert "never-print-this" not in str(exc.value)


def test_project_build_settings_parse(tmp_path):
    path = tmp_path / "nerdit.toml"
    path.write_text(
        '[deploy]\nname="app"\n[deploy.build_settings]\nbuild=false\nsubdir="apps/web"\n'
    )
    assert load_project_config(path).deploy.build_settings == BuildSettings(
        build=False, subdir="apps/web"
    )


def test_node_overrides_supersede_repo_and_safely_encode_commands(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "engines": {"node": "20"},
                "packageManager": "npm@invalid",
                "scripts": {"build": "exit 42", "start": "exit 43"},
            }
        )
    )
    (tmp_path / ".nvmrc").write_text("20")
    install = 'printf "%s" "hello\\world" && npm install'
    build = "node -e 'console.log(\"built\")'"
    start = "node compiled.js"
    plan = detect(
        tmp_path,
        DeployConfig(
            name="app",
            start="old",
            build_settings=BuildSettings(
                install=install,
                build=build,
                start=start,
                node_version="22.23.2",
                package_manager="npm@10.9.9",
            ),
        ),
    )
    assert plan.node_version == "22.23.2"
    assert plan.package_manager == "npm@10.9.9"
    assert (plan.install_command, plan.build_command, plan.effective_start_command) == (
        install,
        build,
        start,
    )
    text = plan.dockerfile_text
    assert text.index("COPY . .") < text.index("RUN " + json.dumps(["sh", "-c", install]))
    assert "RUN " + json.dumps(["sh", "-c", build]) in text
    assert "CMD " + json.dumps(["sh", "-c", start]) in text
    assert "RUN npm run build" not in text


def test_manager_override_still_checks_lock(tmp_path):
    (tmp_path / "package.json").write_text('{"packageManager":"npm@11.19.1"}')
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}')
    with pytest.raises(BuildpackNotSupported, match="conflicts with the lockfile"):
        detect(
            tmp_path,
            DeployConfig(name="app", build_settings=BuildSettings(package_manager="pnpm@10.34.5")),
        )


@pytest.mark.parametrize("framework", ["node", "nextjs"])
def test_build_false_disables_while_none_detects(tmp_path, framework):
    (tmp_path / "package.json").write_text(
        json.dumps(
            {
                "scripts": {"build": "exit 42", "start": "node server.js"},
                "dependencies": {"next": "16.3.4"} if framework == "nextjs" else {},
            }
        )
    )
    disabled = detect(tmp_path, DeployConfig(name="app", build_settings=BuildSettings(build=False)))
    inferred = detect(tmp_path, DeployConfig(name="app", build_settings=BuildSettings(build=None)))
    assert disabled.build_command is None and "RUN npm run build" not in disabled.dockerfile_text
    assert (
        inferred.build_command == "npm run build"
        and "RUN npm run build" in inferred.dockerfile_text
    )


def test_next_explicit_commands_can_replace_absent_scripts(tmp_path):
    (tmp_path / "package.json").write_text('{"dependencies":{"next":"16.3.4"}}')
    plan = detect(
        tmp_path,
        DeployConfig(
            name="app",
            build_settings=BuildSettings(build="npm exec next build", start="npm exec next start"),
        ),
    )
    assert plan.build_command == "npm exec next build"
    assert plan.start_command == "npm exec next start"


def test_dockerfile_wins_with_warning_and_start_override(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch")
    (tmp_path / "package.json").write_text("invalid")
    plan = detect(
        tmp_path,
        DeployConfig(
            name="app", start="old", build_settings=BuildSettings(build=False, start="new")
        ),
    )
    assert plan.dockerfile_text is None and plan.start_command == "new"
    assert plan.build_command is None and plan.warnings


def test_python_custom_steps_and_default_metadata(tmp_path):
    (tmp_path / "requirements.txt").write_text("")
    baseline = detect(tmp_path)
    assert baseline.install_command == "pip install --no-cache-dir -r requirements.txt"
    assert baseline.effective_start_command == "uvicorn main:app --host 0.0.0.0 --port 8000"
    custom = detect(
        tmp_path,
        DeployConfig(
            name="app",
            build_settings=BuildSettings(
                install="pip install .", build="python compile.py", start="python app.py"
            ),
        ),
    )
    assert custom.dockerfile_text.index("COPY . .") < custom.dockerfile_text.index(
        'RUN ["sh", "-c", "pip install ."]'
    )
    assert custom.build_command == "python compile.py" and custom.start_command == "python app.py"
    with pytest.raises(BuildpackNotSupported, match="do not apply to Python"):
        detect(
            tmp_path, DeployConfig(name="app", build_settings=BuildSettings(node_version="24.20.0"))
        )


def test_node_fallback_preview_matches_actual_command(tmp_path):
    (tmp_path / "package.json").write_text('{"packageManager":"yarn@4.18.0"}')
    plan = detect(tmp_path)
    assert plan.start_command is None
    assert plan.effective_start_command == "yarn node server.js"
    assert 'CMD ["yarn", "node", "server.js"]' in plan.dockerfile_text


@pytest.mark.parametrize("field", ["install", "build", "start"])
@pytest.mark.parametrize(
    "command",
    [
        'curl -H "Authorization: Bearer private-sentinel" https://example.com',
        'curl -H "proxy-authorization: Basic private-sentinel" https://example.com',
        "echo Bearer private-sentinel",
        "NPM_TOKEN=private-sentinel npm ci",
        "export AWS_SECRET_ACCESS_KEY=private-sentinel; npm ci",
        "npm ci --token private-sentinel",
        "tool --api-key=private-sentinel",
        "npm config set //registry.example.com/:_authToken private-sentinel",
        "curl https://user:private-sentinel@example.com/file",
        'curl "https://example.com/file?access_token=private-sentinel"',
    ],
)
def test_credential_command_syntax_is_rejected_without_echo(field, command):
    with pytest.raises(ValidationError) as exc:
        BuildSettings.model_validate({field: command})
    assert "private-sentinel" not in str(exc.value)


@pytest.mark.parametrize(
    "command",
    [
        "npm ci && npm run build",
        "NODE_ENV=production npm run build",
        "NODE_OPTIONS=--max-old-space-size=4096 npm run build",
        "npm config set registry https://registry.npmjs.org",
        "python -m pip install -r requirements.txt",
        "node server.js --port 3000",
        "npm run generate-tokens",
        "curl https://example.com/file?version=2",
    ],
)
def test_ordinary_build_commands_remain_allowed(command):
    assert BuildSettings(install=command).install == command
