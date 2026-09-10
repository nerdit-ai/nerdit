"""Verify binaries and public tags share the exact assembled source tree."""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _jobs():
    workflow = ROOT / ".github/workflows/release.yml"
    if not workflow.exists() and not (ROOT / "packaging/RELEASING.md").exists():
        pytest.skip("private release workflow is not mirrored to the public tree")
    yaml = pytest.importorskip("yaml", reason="PyYAML is not installed")
    return yaml.safe_load(workflow.read_text())["jobs"]


def _step(job, name):
    return next(step for step in _jobs()[job]["steps"] if step.get("name") == name)


def test_binary_build_and_publication_consume_one_clean_seed():
    jobs = _jobs()
    assert jobs["publish_source"]["needs"] == "preflight"
    assert jobs["build"]["needs"] == "publish_source"
    assert set(jobs["publish"]["needs"]) == {"build", "publish_source"}
    assert "if" not in jobs["publish_source"]
    for job in ("build", "publish"):
        steps = jobs[job]["steps"]
        assert not any(s.get("uses", "").startswith("actions/checkout@") for s in steps)
        seeds = [
            s
            for s in steps
            if s.get("with", {}).get("name") == "${{ needs.publish_source.outputs.seed_artifact }}"
        ]
        assert len(seeds) == 1
        assert seeds[0]["with"]["path"] == "${{ runner.temp }}/seed-artifact"
        assert "seed-artifact/seed.tar" in _step(job, "Extract the public source")["run"]
    steps = jobs["publish_source"]["steps"]
    names = [s.get("name") for s in steps]
    assert names.index("Pack the validated seed") < names.index("Install the seed standalone")
    assert (
        'tar -cf "$RUNNER_TEMP/seed.tar" -C "$RUNNER_TEMP/seed" .'
        in _step("publish_source", "Pack the validated seed")["run"]
    )


@pytest.mark.parametrize("value", ["", "false", "true"])
def test_source_publication_is_mandatory(value):
    result = subprocess.run(
        ["bash", "-c", _step("preflight", "Require public source publication")["run"]],
        env={**os.environ, "PUBLISH_SOURCE": value},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (value == "true")
    if value != "true":
        assert "PUBLISH_SOURCE=true" in result.stdout


def _git(directory, *args, env=None):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], env=env, text=True, stderr=subprocess.PIPE
    ).strip()


@pytest.mark.parametrize(
    "scenario", ["older_tag", "wrong_tag", "wrong_subject", "fresh", "tag_only"]
)
def test_publish_source_checks_real_git_trees(tmp_path, scenario):
    # URL rewriting keeps the workflow's actual authenticated clone command local.
    remote = tmp_path / "remote.git"
    repo = tmp_path / "author"
    repo.mkdir()
    config = tmp_path / "gitconfig"
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(config),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "TOKEN": "scratch",
        "GH_TOKEN": "scratch",
        "RELEASES_REPO": "test/engine",
        "TAG": "v1.0.0",
        "PUBLISH_AUTHOR_NAME": "Test",
        "PUBLISH_AUTHOR_EMAIL": "test@example.invalid",
        "GITHUB_ENV": str(tmp_path / "github-env"),
    }
    config.write_text(
        f'[url "file://{remote}"]\n'
        "    insteadOf = https://x-access-token:scratch@github.com/test/engine.git\n"
        '[protocol "file"]\n    allow = always\n'
    )
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=main", str(remote)],
        env=env,
        check=True,
        capture_output=True,
    )
    _git(repo, "init", "--initial-branch=main", env=env)
    (repo / "engine.txt").write_text(
        "old\n" if scenario in {"fresh", "wrong_tag", "wrong_subject"} else "release\n"
    )
    _git(repo, "add", ".", env=env)
    _git(repo, "commit", "-m", "initial" if scenario == "fresh" else "nerdit v1.0.0", env=env)
    release_sha = _git(repo, "rev-parse", "HEAD", env=env)
    if scenario in {"older_tag", "wrong_tag", "tag_only"}:
        _git(repo, "tag", "-a", "v1.0.0", "-m", "release", env=env)
    if scenario == "older_tag":
        (repo / "engine.txt").write_text("newer\n")
        _git(repo, "commit", "-am", "nerdit v2.0.0", env=env)
    _git(repo, "remote", "add", "origin", str(remote), env=env)
    _git(repo, "push", "origin", "main", "--tags", env=env)
    previous_head = _git(remote, "rev-parse", "HEAD", env=env)

    seed = tmp_path / "input"
    seed.mkdir()
    (seed / "engine.txt").write_text("release\n")
    runner = tmp_path / "runner"
    artifact = runner / "seed-artifact"
    artifact.mkdir(parents=True)
    with tarfile.open(artifact / "seed.tar", "w") as archive:
        archive.add(seed, arcname=".")
    env["RUNNER_TEMP"] = str(runner)
    result = subprocess.run(
        ["bash", "-c", _step("publish", "Publish source tree")["run"]],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if scenario in {"wrong_tag", "wrong_subject"}:
        assert result.returncode != 0
        assert "differs from the binary build input" in result.stdout
        assert not Path(env["GITHUB_ENV"]).exists()
        assert _git(remote, "rev-parse", "HEAD", env=env) == previous_head
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        sha = Path(env["GITHUB_ENV"]).read_text().strip().removeprefix("SOURCE_SHA=")
        assert _git(remote, "show", f"{sha}:engine.txt", env=env) == "release"
        if scenario == "fresh":
            assert sha != previous_head
            assert _git(remote, "rev-parse", "HEAD", env=env) == sha
            assert _git(remote, "tag", env=env) == ""
        else:
            assert sha == release_sha
            assert _git(remote, "rev-parse", "HEAD", env=env) == previous_head
