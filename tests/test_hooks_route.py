"""The webhook routes, and `[hooks.*]` at deploy, through the real stack.

The `test_apply_route` rig: a real in-memory database under the real routers
and the real auth + audit middleware.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nerdit.core.hooks import HOOK_ID_RE, default_secret_ref
from nerdit.daemon.bootstrap import build_edge_auth_resolver
from nerdit.daemon.routes.hooks import router as hooks_router
from nerdit.db.models import Job, JobKind, JobStatus
from tests.test_apply_route import A_RAW, B_RAW, LEGACY, RO_RAW, _auth, _zip
from tests.test_apply_route import env as env  # noqa: F401
from tests.test_project_delegation import delegated as delegated  # noqa: F401

STRIPE = {"name": "stripe", "sender": "stripe", "path": "/webhooks/stripe"}
TOML = """
[hooks.stripe]
sender = "stripe"
path = "/webhooks/stripe"
"""


@pytest.fixture
async def rig(env):  # noqa: F811
    env.app.include_router(hooks_router)
    resp = await _deploy(env)
    assert resp.status_code == 201, resp.text
    return env


async def _deploy(env, toml: str | None = None, *, dry_run: bool = False, name: str = "shop"):  # noqa: F811
    return await env.client.post(
        "/deploy",
        files={"archive": ("a.zip", _zip(toml), "application/zip")},
        data={"name": name},
        params={"dry_run": "true"} if dry_run else None,
        headers=_auth(A_RAW),
    )


async def _add(env, raw: str = A_RAW, *, key: str | None = "k1", app: str = "shop", **body):  # noqa: F811
    headers = _auth(raw) | ({"Idempotency-Key": key} if key else {})
    return await env.client.post(f"/services/{app}/hooks", json=body or STRIPE, headers=headers)


def _link(env, slug: str | None = "box", domain: str | None = "nodes.example") -> None:  # noqa: F811
    env.app.state.settings.link = SimpleNamespace(slug=slug, nodes_base_domain=domain)


async def _hook_ids(env) -> list[tuple[str, str, str | None]]:  # noqa: F811
    cursor = await env.db.conn.execute(
        "SELECT hook_id, path, revoked_at FROM hooks ORDER BY created_at, hook_id"
    )
    return [tuple(r) for r in await cursor.fetchall()]


async def test_list_hooks_requires_owner_or_admin_and_returns_url_and_secret_key(rig):
    _link(rig)
    added = await _add(rig)
    assert added.status_code == 200, added.text
    body = added.json()
    hook_id = body["url"].rsplit("/", 1)[1]
    assert HOOK_ID_RE.fullmatch(hook_id)
    assert body["url"] == f"https://shop--box.nodes.example/.nerdit/hooks/{hook_id}"
    assert body["secret_key"] == "NERDIT_HOOK_STRIPE_SECRET"
    assert body["secret_ref"] == "${secrets.NERDIT_HOOK_STRIPE_SECRET}"
    assert body["created"] is True

    for raw in (A_RAW, LEGACY):
        listed = await rig.client.get("/services/shop/hooks", headers=_auth(raw))
        assert listed.status_code == 200, listed.text
        assert [h["url"] for h in listed.json()["hooks"]] == [body["url"]]
    foreign = await rig.client.get("/services/shop/hooks", headers=_auth(B_RAW))
    assert foreign.status_code == 403 and hook_id not in foreign.text
    missing = await rig.client.get("/services/nope/hooks", headers=_auth(A_RAW))
    assert missing.status_code == 404


async def test_list_hooks_url_is_null_on_an_unlinked_node(rig):
    assert (await _add(rig)).json()["url"] is None
    _link(rig, slug=None)
    listed = await rig.client.get("/services/shop/hooks", headers=_auth(A_RAW))
    assert listed.json()["hooks"][0]["url"] is None


async def test_add_hook_requires_an_idempotency_key(rig):
    resp = await _add(rig, key=None)
    assert resp.status_code == 400
    assert resp.json()["code"] == "idempotency_key_required"
    assert await _hook_ids(rig) == []


async def test_add_hook_replay_returns_created_false_and_conflict_is_409(rig):
    first = (await _add(rig)).json()
    again = await _add(rig, key="k2")
    assert again.status_code == 200 and again.json()["created"] is False
    assert again.json()["created_at"] == first["created_at"]
    conflict = await _add(rig, key="k3", **{**STRIPE, "path": "/other"})
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "hook.exists"
    assert len(await _hook_ids(rig)) == 1


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({**STRIPE, "name": "Bad_Name"}, "name"),
        ({**STRIPE, "sender": "paypal"}, "shape"),
        ({**STRIPE, "path": "no-slash"}, "shape"),
        ({**STRIPE, "sender": "hmac"}, "shape"),  # hmac without header
        ({**STRIPE, "header": "x-sig"}, "shape"),  # header without hmac
        ({**STRIPE, "sender": "hmac", "header": "Authorization"}, "shape"),
    ],
)
async def test_add_hook_invalid_is_422_value_free(rig, body, reason):
    resp = await _add(rig, **body)
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "hook.invalid"
    assert resp.json()["reason"] == reason
    assert "paypal" not in resp.text and "no-slash" not in resp.text
    assert await _hook_ids(rig) == []


async def test_add_hook_refuses_a_model_or_database_with_422(rig):
    for kind in (JobKind.model, JobKind.database):
        await rig.queries.create_job(
            Job(
                id=f"job-{kind.value}",
                name=f"x-{kind.value}",
                service_name=f"x-{kind.value}",
                kind=kind,
                gpu_count=0,
                status=JobStatus.running,
                submitted_by_token="tok-a",
                config="{}",
            )
        )
        resp = await _add(rig, app=f"x-{kind.value}")
        assert resp.status_code == 422, resp.text
        assert resp.json()["code"] == "hook.kind_unsupported"


async def test_add_hook_is_audited_hook_created_without_the_hook_id(rig):
    _link(rig)
    hook_id = (await _add(rig)).json()["url"].rsplit("/", 1)[1]
    await _add(rig, key="k2", name="Bad_Name", sender="whsec_pasted", path="/x")
    rows = await rig.audit("hook.created")
    assert [r[0] for r in rows] == ["shop", "shop"]
    assert rows[0][1] == {
        "service": "shop",
        "name": "stripe",
        "sender": "stripe",
        "path": "/webhooks/stripe",
    }
    assert rows[1][1] == {"service": "shop", "name": "<refused>", "sender": "<refused>"}
    cursor = await rig.db.conn.execute("SELECT * FROM audit_log")
    dump = json.dumps([list(r) for r in await cursor.fetchall()], default=str)
    assert hook_id not in dump and "whsec_pasted" not in dump


async def test_readonly_and_foreign_owner_tokens_are_refused(rig):
    for raw in (RO_RAW, B_RAW):
        assert (await _add(rig, raw)).status_code == 403
        deleted = await rig.client.delete(
            "/services/shop/hooks/stripe", headers=_auth(raw) | {"Idempotency-Key": "d"}
        )
        assert deleted.status_code == 403
    listed = await rig.client.get("/services/shop/hooks", headers=_auth(RO_RAW))
    assert listed.status_code == 403
    assert await _hook_ids(rig) == []


@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_project_delegated_principal_is_refused_before_the_route(delegated, method):  # noqa: F811
    d = delegated
    resp = await d.client.request(
        method, "/api/services/asso/hooks", json=STRIPE, headers=d.headers
    )
    assert resp.status_code == 403, resp.text
    assert resp.json()["code"] == "project.delegation_forbidden"


async def test_remove_hook_is_idempotent_and_audited_hook_removed(rig):
    await _add(rig)
    for expected in (True, False):
        resp = await rig.client.delete("/services/shop/hooks/stripe", headers=_auth(A_RAW))
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"service_name": "shop", "name": "stripe", "removed": expected}
    odd = await rig.client.delete("/services/shop/hooks/Not_A_Name", headers=_auth(A_RAW))
    assert odd.json() == {"service_name": "shop", "name": "<refused>", "removed": False}
    rows = await rig.audit("hook.removed")
    assert [r[1]["name"] for r in rows] == ["stripe", "stripe", "<refused>"]
    assert [r[2] for r in await _hook_ids(rig)][0] is not None  # revoked, kept


async def test_secret_set_via_the_secrets_route_under_secret_key_is_what_the_gate_resolves(rig):
    key = (await _add(rig)).json()["secret_key"]
    value = "whsec_from_the_provider"  # noqa: S105 - a test literal
    resp = await rig.client.post(
        "/secrets/shop", json={"values": {key: value}}, headers=_auth(A_RAW)
    )
    assert resp.status_code == 200, resp.text
    assert value not in resp.text
    resolve = build_edge_auth_resolver(rig.app.state.secret_manager)
    assert resolve("shop", None, default_secret_ref("stripe")) == value


async def test_deploy_with_hooks_section_upserts_rows_and_dry_run_writes_none(env):  # noqa: F811
    env.app.include_router(hooks_router)
    dry = await _deploy(env, TOML, dry_run=True)
    assert dry.status_code == 200, dry.text
    assert await _hook_ids(env) == []
    real = await _deploy(env, TOML)
    assert real.status_code == 201, real.text
    listed = (await env.client.get("/services/shop/hooks", headers=_auth(A_RAW))).json()
    assert [(h["name"], h["sender"], h["secret_key"]) for h in listed["hooks"]] == [
        ("stripe", "stripe", "NERDIT_HOOK_STRIPE_SECRET")
    ]
    rows = await env.audit("deploy.create")
    assert rows[-1][1]["hooks"] == ["stripe"]


async def test_redeploy_with_a_changed_path_keeps_the_hook_id(env):  # noqa: F811
    assert (await _deploy(env, TOML)).status_code == 201
    (before,) = await _hook_ids(env)
    resp = await _deploy(env, TOML.replace("/webhooks/stripe", "/hooks/v2"))
    assert resp.status_code == 201, resp.text
    (after,) = await _hook_ids(env)
    assert after[0] == before[0] and after[1] == "/hooks/v2" and after[2] is None


async def test_deploy_with_invalid_hooks_section_is_422_deploy_invalid_hooks_without_the_literal(
    env,  # noqa: F811
):
    literal = "whsec_literal_in_the_file"  # noqa: S105 - a test literal
    resp = await _deploy(env, TOML + f'secret = "{literal}"\n')
    assert resp.status_code == 422, resp.text
    assert resp.json()["code"] == "deploy.invalid_hooks"
    assert literal not in resp.text
    assert await _hook_ids(env) == []
