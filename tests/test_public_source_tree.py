"""Exercise the public assembler against a committed miniature repository."""

import os
import shutil
import subprocess
from pathlib import Path


def test_public_source_keeps_tests_and_excludes_website_docs(tmp_path):
    source = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
        )

    files = {
        "src/nerdit/__init__.py": "",
        "tests/test_example.py": "def test_example(): assert True\n",
        "docs/guide/install.md": "Website-only installation guide\n",
        "mkdocs.yml": "site_name: Nerdit\n",
        "docker/Dockerfile": "FROM scratch\n",
        "examples/README.md": "Example\n",
        "pyproject.toml": '[project]\nname = "nerdit"\n',
        "nerdit.toml.example": "",
        ".pre-commit-config.yaml": "repos: []\n",
    }
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    overlay = source / "packaging/public"
    if not overlay.is_dir():
        overlay = source  # Public checkouts already have the overlaid root files.
    target = repo / "packaging/public"
    target.mkdir(parents=True)
    for name in (
        "Makefile",
        "README.md",
        "CONTRIBUTING.md",
        "CLAUDE.md",
        "LICENSE",
        "NOTICE",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "CHANGELOG.md",
        ".gitignore",
    ):
        shutil.copy2(overlay / name, target / name)
    shutil.copytree(overlay / ".github", target / ".github")
    (repo / "scripts").mkdir()
    for path in (source / "scripts").iterdir():
        if path.is_file():
            shutil.copy2(path, repo / "scripts" / path.name)
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-qm", "Fixture")

    out = tmp_path / "public"
    subprocess.run(
        ["bash", "scripts/publish_public.sh", "--out", str(out)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not (out / "docs").exists()
    assert not (out / "mkdocs.yml").exists()
    assert (out / "tests/test_example.py").is_file()
    for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md", "LICENSE", "NOTICE"):
        assert (out / name).is_file()
    assert "https://nerdit.ai/" in (out / "README.md").read_text()
    assert "](docs/" not in (out / "README.md").read_text()
    assert "mkdocs" not in (out / "Makefile").read_text()
    assert "pytest" in (out / "Makefile").read_text()

    # Excluding the website must not weaken the private-reference gate.
    (repo / "src/nerdit/__init__.py").write_text("# " + "tasks" + "/private-plan.md\n")
    git("add", ".")
    git("commit", "-qm", "Forbidden reference")
    rejected = tmp_path / "rejected"
    result = subprocess.run(
        ["bash", "scripts/publish_public.sh", "--out", str(rejected)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "forbidden private reference" in result.stderr
    assert not rejected.exists()
