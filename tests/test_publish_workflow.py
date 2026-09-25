"""Pin the release workflow's PyPI identity, isolation and artifact contents.

Trusted Publishing is bound to the private repository, publish.yml and any
matching environment. Renaming or moving the upload breaks OIDC. Only the publish
job gets id-token: write and only downloads artifacts/runs the PyPA action;
checkout, npm and pip stay in the unprivileged build job. Never add stored tokens.

Require PUBLISH_SOURCE and an explicit tagged checkout. Build the dashboard before
Python artifacts, inspect both for the bundle, prepare the public README before
building, and reject private-repository references in wheel metadata.
These static guards do not replace publishing a real tag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish.yml"
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The private development repository's name, assembled from two halves rather
#: than written out. ``scripts/publish_public.sh`` ships ``tests/`` whole and
#: ends with a gate that aborts the publish on that literal appearing anywhere
#: in the seed — so a test *about* the private repo must not be its own match.
#: Same trick the assembler itself uses for the same reason.
PRIVATE_REPO_NAME = "nerdit" + "_private"

#: The public seed carries neither ``.github/workflows/`` from the private tree
#: (``scripts/publish_public.sh`` overlays ``packaging/public/.github`` instead)
#: nor ``packaging/RELEASING.md``. The freeze test tells the two trees apart the
#: same way: absent both, this is the public tree and there is nothing to pin.
PRIVATE_MARKER = REPO_ROOT / "packaging" / "RELEASING.md"


def _require_private_tree() -> None:
    if not WORKFLOW.is_file() and not PRIVATE_MARKER.is_file():
        pytest.skip(
            "public source tree: the release/publish workflows are not mirrored, "
            "so there is nothing to pin"
        )


def _text() -> str:
    _require_private_tree()
    assert WORKFLOW.is_file(), (
        ".github/workflows/publish.yml is missing — the PyPI Trusted Publisher "
        "is bound to that exact filename in this repo; renaming or removing it "
        "breaks the OIDC exchange (open-source plan §3 Q7)."
    )
    return WORKFLOW.read_text(encoding="utf-8")


def _code() -> str:
    """``publish.yml`` with every whole-line comment stripped.

    The header block *explains* the rails it enforces — it names ``secrets.*``
    and ``hatch build`` in prose — so a substring search over the raw file
    answers a question about the documentation, not about what runs. Every
    assertion about behaviour reads this instead. Same trick as
    ``test_packaging_freeze.py::test_smoke_never_uses_the_forbidden_teardowns``.
    """
    return "\n".join(line for line in _text().splitlines() if not line.lstrip().startswith("#"))


def _doc() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml", reason="PyYAML is not installed")
    parsed = yaml.safe_load(_text())
    assert isinstance(parsed, dict)
    return parsed


def _triggers(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the ``on:`` block.

    YAML 1.1 (which PyYAML implements) resolves the bare key ``on`` to the
    boolean ``True``, so the key is looked up both ways.
    """
    block = doc.get("on", doc.get(True))
    assert isinstance(block, dict), "publish.yml has no mapping-shaped `on:` block"
    return block


def _jobs() -> dict[str, Any]:
    jobs = _doc()["jobs"]
    assert list(jobs) == ["build", "publish"], (
        "publish.yml must be exactly two jobs, `build` (no OIDC) then `publish` "
        f"(OIDC, nothing else) — got {list(jobs)}"
    )
    return dict(jobs)


def _build_job() -> dict[str, Any]:
    return dict(_jobs()["build"])


def _publish_job() -> dict[str, Any]:
    return dict(_jobs()["publish"])


def test_publish_workflow_exists_and_is_named() -> None:
    assert _doc()["name"] == "Publish to PyPI"


def test_triggers_are_release_completion_and_dispatch() -> None:
    triggers = _triggers(_doc())
    assert set(triggers) == {"workflow_run", "workflow_dispatch"}, (
        "publish.yml must run only after the Release workflow or on an explicit "
        f"dispatch; found triggers {sorted(map(str, triggers))}"
    )
    run = triggers["workflow_run"]
    assert run["workflows"] == ["Release"], (
        f"publish.yml must chain off the Release workflow by its `name:` — got {run['workflows']!r}"
    )
    assert run["types"] == ["completed"]

    dispatch = triggers["workflow_dispatch"]
    tag = dispatch["inputs"]["tag"]
    assert tag["required"] is True, "the dispatch `tag` input must be required"


def test_no_pull_request_trigger() -> None:
    """A fork PR must never be able to run the job that holds the OIDC grant."""
    triggers = _triggers(_doc())
    for forbidden in ("pull_request", "pull_request_target"):
        assert forbidden not in triggers, f"publish.yml must not trigger on {forbidden}"


def test_only_the_publish_job_holds_the_oidc_grant() -> None:
    doc = _doc()
    jobs = _jobs()

    assert jobs["build"]["permissions"] == {"contents": "read"}, (
        "the build job runs `npm ci`, `pip install` and a checkout — it must "
        "hold `contents: read` and NOTHING else, so a compromised dependency "
        "cannot reach an OIDC token: "
        f"{jobs['build'].get('permissions')!r}"
    )
    assert jobs["publish"]["permissions"] == {"contents": "read", "id-token": "write"}, (
        "the publish job needs exactly `contents: read` and `id-token: write` "
        "(the Trusted Publisher exchange) — nothing else"
    )
    # A workflow-level default is allowed, but it must not widen anything.
    workflow_permissions = doc.get("permissions")
    if workflow_permissions is not None:
        assert workflow_permissions == {"contents": "read"}, workflow_permissions

    holders = [name for name, job in jobs.items() if "id-token" in (job.get("permissions") or {})]
    assert holders == ["publish"], f"exactly one job may request id-token; got {holders}"


def test_the_publish_job_does_nothing_but_download_and_upload() -> None:
    """Anything else in that job runs beside a live PyPI publish grant."""
    job = _publish_job()
    assert job["needs"] == "build" or job["needs"] == ["build"], job.get("needs")

    steps = job["steps"]
    uses = [str(step.get("uses", "")) for step in steps]
    assert all("run" not in step for step in steps), (
        "the publish job must run no shell at all — every script step belongs "
        f"in `build`: {[s.get('name') or s.get('uses') for s in steps if 'run' in s]}"
    )
    assert len(steps) == 2, f"expected exactly two steps in the publish job, got {uses}"
    assert uses[0].startswith("actions/download-artifact@"), uses
    assert uses[1].startswith("pypa/gh-action-pypi-publish@"), uses


def test_publish_job_is_gated_on_the_resolve_skip() -> None:
    condition = str(_publish_job()["if"])
    assert "needs.build.outputs.skip" in condition, (
        "a Release run that carries no release tag must end as a clean skip, "
        "so the publish job has to consult the resolve step's `skip` output"
    )


def test_build_job_is_gated_on_publish_source() -> None:
    condition = _build_job()["if"]
    assert "vars.PUBLISH_SOURCE == 'true'" in condition, (
        "a PyPI upload is as irreversible as the source push and must ride the "
        "same owner switch; the job `if` no longer checks PUBLISH_SOURCE"
    )
    assert "workflow_run.conclusion == 'success'" in condition, (
        "a failed Release must not publish to PyPI"
    )


def test_no_stored_credentials() -> None:
    """OIDC only — a `secrets.` reference means a stored API token came back."""
    code = _code()
    assert "secrets." not in code, (
        "publish.yml must reference no repository secret: the Trusted Publisher "
        "grant is minted from `id-token: write` alone"
    )
    assert "password:" not in code, (
        "pypa/gh-action-pypi-publish must be called without a `password:` — "
        "supplying one disables the OIDC path"
    )


def _resolve_step() -> dict[str, Any]:
    for step in _build_job()["steps"]:
        if "id" in step and "GITHUB_OUTPUT" in str(step.get("run", "")):
            return dict(step)
    raise AssertionError("publish.yml has no tag-resolution step writing to GITHUB_OUTPUT")


def test_checks_out_the_tag_not_the_default_branch() -> None:
    """`workflow_run` defaults to the default branch; that would ship master."""
    assert "refs/tags/" in _code(), (
        "publish.yml must check out refs/tags/<tag> explicitly — a workflow_run "
        "checkout otherwise lands on the default branch"
    )
    expected = "${{ steps.%s.outputs.ref }}" % _resolve_step()["id"]
    steps = _build_job()["steps"]
    checkouts = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert checkouts, "publish.yml never checks the repository out"
    for step in checkouts:
        assert step["with"]["repository"] == "nerdit-ai/nerdit"
        ref = str(step.get("with", {}).get("ref", ""))
        assert ref == expected, (
            "the checkout must use the validated ref the resolve step emitted, "
            f"verbatim — expected {expected!r}, got {ref!r}"
        )


def test_a_release_run_without_a_tag_skips_instead_of_failing() -> None:
    """`release.yml` is dispatchable, and then head_branch is a BRANCH.

    Its head_sha is the branch tip, not the tag the Release built, so the
    resolver must never guess a tag from it (Codex review on PR #152): a
    dispatched Release is a clean green skip, and the operator dispatches
    this workflow with the tag by hand.
    """
    step = _resolve_step()
    run = str(step["run"])
    env_keys = {str(k).lower() for k in (step.get("env") or {})}
    assert "head_sha" not in env_keys and "HEAD_SHA" not in run, (
        "head_sha is the dispatch branch's tip, not the built tag — the resolver must not read it"
    )
    assert "matching-refs" not in run and "git/tags/" not in run, (
        "no tag lookup by commit: guessing a tag from head_sha can publish the "
        "wrong version, and PyPI cannot take it back"
    )
    assert "skip=true" in run and "::notice::" in run, (
        "a Release run with no release tag must emit a ::notice:: and set "
        "skip=true — a clean green skip, never a failed release"
    )
    assert "RUN_EVENT" in str(step.get("env") or {}) and '= "push" ]' in run, (
        "head_branch may only be trusted as a tag when the Release run was a "
        "tag PUSH: a Release dispatched from a branch named like v1.2.3 also "
        "matches the regex while having built the tag its input named"
    )


def test_version_coherence_is_enforced() -> None:
    """Mirrors release.yml: the tag, pyproject and `nerdit.__version__` agree."""
    code = _code()
    assert "['project']['version']" in code or '["project"]["version"]' in code, (
        "publish.yml must compare the tag against pyproject [project].version"
    )
    assert "nerdit.__version__" in code, (
        "publish.yml must compare the tag against nerdit.__version__ too — "
        "that is the literal the daemon reports at runtime"
    )
    assert "nerdit --version" in code, (
        "the wheel must be installed into a throwaway venv and its CLI asked "
        "for the version — that is what a customer actually runs"
    )


def test_build_order_dashboard_then_hatch_then_upload() -> None:
    code = _code()
    npm = code.index("npm run build")
    readme = code.index("Prepare the public README for PyPI")
    hatch = code.index("hatch build")
    upload = code.index("pypa/gh-action-pypi-publish")
    assert npm < hatch, (
        "the dashboard bundle is not committed, so `npm run build` must precede "
        "`hatch build` or the wheel ships without the SPA"
    )
    assert readme < hatch, (
        "the public README links must be prepared BEFORE `hatch build` — the long "
        "description is baked into the wheel's METADATA at build time"
    )
    assert hatch < upload, "the distribution must be built before it is uploaded"


def test_asserts_the_dashboard_bundle_in_both_artifacts() -> None:
    code = _code()
    assert "src/nerdit/daemon/web/dist/index.html" in code, (
        "publish.yml must assert the dashboard bundle exists after npm run build "
        "(the same guard packaging/nerdit.spec carries)"
    )
    assert "nerdit/daemon/web/dist/index.html" in code and "unzip -l" in code, (
        "publish.yml must verify the BUILT WHEEL contains "
        "nerdit/daemon/web/dist/index.html — a UI-less wheel on PyPI cannot be "
        "replaced, only yanked"
    )
    assert "tar tzf" in code, (
        "the sdist must be checked too: an sdist missing the dashboard builds "
        "a UI-less wheel for anyone who builds from source"
    )
    assert "twine check dist/*" in code


def test_build_requires_a_published_public_release() -> None:
    step = _resolve_step()
    assert step["env"]["REPO"] == "nerdit-ai/nerdit"
    run = step["run"]
    assert "repos/$REPO/releases/tags/$RAW_TAG" in run
    assert ".draft == false" in run
    assert '"$PUBLISHED" != "true"' in run
    assert "repos/$REPO/git/ref/tags/$RAW_TAG" in run


def test_the_long_description_is_the_public_readme() -> None:
    code = _code()
    assert "cp packaging/public/README.md README.md" not in code
    assert "Prepare the public README for PyPI" in code


def test_the_long_description_has_no_relative_links() -> None:
    """PyPI renders the README under pypi.org/project/nerdit/, so relative
    links (docs/guide/install.md, LICENSE, CONTRIBUTING.md) 404 there. The
    overlay step rewrites them to the public repo and fails if any survive."""
    code = _code()
    assert "https://github.com/nerdit-ai/nerdit/blob/{os.environ['TAG']}/" in code, (
        "relative README links must be rewritten to the public repository"
    )
    assert "relative links survived the rewrite" in code, (
        "the rewrite must assert that no relative link is left behind"
    )


def test_the_wheel_metadata_is_checked_for_the_private_repo_name() -> None:
    """If the overlay ever no-ops, PyPI would show the private repo forever."""
    code = _code()
    assert "dist-info/METADATA" in code and "unzip -p" in code, (
        "publish.yml must read the built wheel's own METADATA back — that text "
        "IS the PyPI project page, and the version can never be re-uploaded"
    )
    assert PRIVATE_REPO_NAME in _text().replace('""', ""), (
        "the METADATA check must actually search for the private repository's "
        "name (assembled from halves in the shell, as publish_public.sh does)"
    )


def test_pinned_action_versions() -> None:
    """Same pinning convention as release.yml: major tags, no floating refs."""
    uses = [str(s["uses"]) for job in _jobs().values() for s in job["steps"] if "uses" in s]
    assert "actions/checkout@v4" in uses
    assert "actions/setup-python@v5" in uses
    assert "actions/setup-node@v4" in uses
    assert "actions/upload-artifact@v4" in uses
    assert "actions/download-artifact@v4" in uses
    # PyPA's own documented mutable ref for this action; pinning a SHA here
    # would silently miss the security fixes they ship through it.
    assert "pypa/gh-action-pypi-publish@release/v1" in uses


def test_the_upload_tolerates_an_already_uploaded_file() -> None:
    """The re-run story: dispatch the tag; a half-uploaded pair completes."""
    publish_step = _publish_job()["steps"][-1]
    assert publish_step.get("with", {}).get("skip-existing") is True, (
        "pypa/gh-action-pypi-publish must be called with `skip-existing: true` "
        "so a re-dispatch after a partial upload completes the pair instead of "
        "failing on the file PyPI already accepted"
    )


def test_the_artifact_name_matches_between_the_two_jobs() -> None:
    upload = next(
        s
        for s in _build_job()["steps"]
        if str(s.get("uses", "")).startswith("actions/upload-artifact@")
    )
    download = next(
        s
        for s in _publish_job()["steps"]
        if str(s.get("uses", "")).startswith("actions/download-artifact@")
    )
    assert upload["with"]["name"] == download["with"]["name"], (
        "the publish job downloads the artifact the build job uploaded; a name "
        "mismatch fails at download time, after the release already shipped"
    )
    assert download["with"]["path"] == "dist"


def test_pyproject_urls_point_at_the_public_repo() -> None:
    """The PyPI page must send visitors to the mirror, not the private repo."""
    import tomllib

    with PYPROJECT.open("rb") as fh:
        project = tomllib.load(fh)["project"]

    urls = project["urls"]
    for key in ("Homepage", "Repository", "Issues", "Changelog"):
        assert key in urls, f"pyproject [project.urls] is missing {key}"
        assert urls[key].startswith("https://github.com/nerdit-ai/nerdit"), (
            f"[project.urls] {key} must point at the public mirror: {urls[key]!r}"
        )
    assert urls["Issues"].endswith("/issues")
    assert urls["Changelog"].endswith("/blob/main/CHANGELOG.md")
    assert PRIVATE_REPO_NAME not in str(urls), (
        "no PyPI-visible URL may point at the private development repo"
    )


def test_pyproject_license_metadata_is_pep639_only() -> None:
    """`license` expression + classifiers, never a deprecated License:: pair."""
    import tomllib

    with PYPROJECT.open("rb") as fh:
        project = tomllib.load(fh)["project"]

    assert project["license"] == "Apache-2.0", project["license"]
    classifiers = project["classifiers"]
    assert not [c for c in classifiers if c.startswith("License ::")], (
        "PEP 639 deprecates the License:: classifiers in favour of the `license` "
        "expression, and hatchling rejects carrying both"
    )
    for expected in (
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ):
        assert expected in classifiers, f"missing classifier {expected!r}"


def test_release_workflow_is_still_named_release() -> None:
    """publish.yml chains off `workflows: ["Release"]` — by name, not filename."""
    _require_private_tree()
    if not RELEASE_WORKFLOW.is_file():
        pytest.skip("release.yml is absent from this tree")
    assert "\nname: Release\n" in RELEASE_WORKFLOW.read_text(encoding="utf-8"), (
        "release.yml's `name:` is the string publish.yml's workflow_run trigger "
        "matches on; renaming it silently stops every PyPI publish"
    )


def test_the_sdist_is_an_allowlist() -> None:
    """Hatchling's default sdist is every tracked file. In the private tree that
    is the planning layer, the runbooks, the hosting layout and the CI files —
    everything publish_public.sh keeps out of the mirror — inside a tarball
    PyPI never deletes (found by the /security-review on PR #152)."""
    import tomllib

    with PYPROJECT.open("rb") as fh:
        hatch = tomllib.load(fh)["tool"]["hatch"]["build"]

    sdist = hatch.get("targets", {}).get("sdist", {})
    assert set(sdist.get("only-include", [])) == {
        "src/nerdit",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "NOTICE",
    }, "[tool.hatch.build.targets.sdist] only-include must pin the package plus its four root files"


def test_the_workflow_requires_public_license_and_notice() -> None:
    code = _code()
    assert "for f in README.md LICENSE NOTICE" in code
    assert 'test -f "$f"' in code
    assert "cp packaging/public/" not in code


def test_the_sdist_contents_are_checked_against_the_allowlist() -> None:
    """A pyproject edit must not be able to widen the sdist unnoticed: the
    built tarball is re-read member by member and refused on anything else."""
    code = _code()
    assert "Refuse an sdist that carries anything but the package" in code
    assert (
        "tar tzf" in code
        and "src/nerdit/*|pyproject.toml|README.md|LICENSE|NOTICE|PKG-INFO" in code
    ), "the sdist gate must enumerate the allowlist the way pyproject does"
    assert "tar xzOf" in code, "the sdist's text must be searched for the private repo name too"
    assert code.index("Refuse an sdist") < code.index("upload-artifact"), (
        "the sdist gate runs before the artifact leaves the build job"
    )


def test_standard_install_includes_mcp_with_compatible_extra() -> None:
    import tomllib

    project = tomllib.loads(PYPROJECT.read_text())["project"]
    assert "mcp>=1.9,<2" in project["dependencies"]
    assert project["optional-dependencies"]["mcp"] == ["mcp>=1.9,<2"]
