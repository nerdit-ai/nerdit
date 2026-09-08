"""Pin secret-reference precedence, loader calls and exact error messages.

Exercise the real AI and database resolvers. `model_construct` bypasses config
validation to reach defensive invalid-reference checks. Per-service values win;
unscoped references never fall back to shared secrets. Test `walk_secret_ref`
separately to preserve loader selection and failure propagation.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from nerdit.config.project import AiBindingConfig, DbBindingConfig
from nerdit.core.bindings.secretref import walk_secret_ref
from nerdit.core.data.binding import _resolve_external
from nerdit.core.models.binding import BindingNotReady, _resolve_api
from nerdit.core.secrets import SecretDecryptError


def _ai_cfg(api_key: str | None) -> AiBindingConfig:
    return AiBindingConfig.model_construct(
        provider="api",
        model="gpt-4o-mini",
        base_url="https://api.example.com/v1",
        api_key=api_key,
    )


def _db_cfg(
    password: str | None, url: str = "postgresql://appuser@db.example.com:5432/app"
) -> DbBindingConfig:
    return DbBindingConfig.model_construct(provider="external", url=url, password=password)


# --- (a) invalid ref (defensive; unreachable via the normal parse grammar) ------


def test_ai_invalid_ref_message():
    with pytest.raises(BindingNotReady) as exc:
        _resolve_api("X", _ai_cfg("not-a-ref"), {}, {})
    assert str(exc.value) == (
        "[ai.X] api_key is not a ${secrets.KEY} reference; "
        "redeploy the app with a valid [ai.X] section"
    )


def test_db_invalid_ref_message():
    with pytest.raises(BindingNotReady) as exc:
        _resolve_external("X", _db_cfg("not-a-ref"), {}, {})
    assert str(exc.value) == (
        "[db.X] password is not a ${secrets.KEY} reference; "
        "redeploy the app with a valid [db.X] section"
    )


# --- (b) unset shared secret -----------------------------------------------------


def test_ai_shared_secret_not_set_message():
    with pytest.raises(BindingNotReady) as exc:
        _resolve_api("X", _ai_cfg("${secrets.shared.K}"), {}, {})
    assert str(exc.value) == (
        "[ai.X] shared secret 'K' not set — run: nerdit secrets set --shared K=..."
    )


def test_db_shared_secret_not_set_message():
    with pytest.raises(BindingNotReady) as exc:
        _resolve_external("X", _db_cfg("${secrets.shared.K}"), {}, {})
    assert str(exc.value) == (
        "[db.X] shared secret 'K' not set — run: nerdit secrets set --shared K=..."
    )


# --- (c) unset per-service secret ------------------------------------------------


def test_ai_per_service_secret_not_set_message():
    with pytest.raises(BindingNotReady) as exc:
        _resolve_api("X", _ai_cfg("${secrets.K}"), {}, {})
    assert str(exc.value) == "[ai.X] secret 'K' not set — run: nerdit secrets set <app> K=..."


def test_db_per_service_secret_not_set_message():
    with pytest.raises(BindingNotReady) as exc:
        _resolve_external("X", _db_cfg("${secrets.K}"), {}, {})
    assert str(exc.value) == "[db.X] secret 'K' not set — run: nerdit secrets set <app> K=..."


# --- (d) precedence: per-service wins over shared; unscoped never falls back -----


def test_ai_shared_ref_per_service_value_wins_over_shared():
    resolved = _resolve_api("X", _ai_cfg("${secrets.shared.K}"), {"K": "own"}, {"K": "shared"})
    assert resolved.api_key == "own"


def test_db_shared_ref_per_service_value_wins_over_shared():
    resolved = _resolve_external("X", _db_cfg("${secrets.shared.K}"), {"K": "own"}, {"K": "shared"})
    assert ":own@" in resolved.url


def test_ai_unscoped_ref_never_falls_back_to_shared():
    with pytest.raises(BindingNotReady):
        _resolve_api("X", _ai_cfg("${secrets.K}"), {}, {"K": "shared"})


def test_db_unscoped_ref_never_falls_back_to_shared():
    with pytest.raises(BindingNotReady):
        _resolve_external("X", _db_cfg("${secrets.K}"), {}, {"K": "shared"})


# --- (e) the kernel: precedence matrix + loader-call discipline ------------------


class _Loader:
    """A dict-backed scope loader that records how often it was consulted."""

    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.calls = 0

    def __call__(self) -> Mapping[str, str]:
        self.calls += 1
        return self.values


def test_walk_unscoped_ref_resolves_per_service():
    service, shared = _Loader({"K": "own"}), _Loader({"K": "global"})
    res = walk_secret_ref("${secrets.K}", service_env=service, shared_env=shared)
    assert (res.matched, res.scope, res.key) == (True, None, "K")
    assert (res.value, res.source) == ("own", "service")
    assert shared.calls == 0


def test_walk_unscoped_ref_never_falls_back_to_shared():
    service, shared = _Loader({}), _Loader({"K": "global"})
    res = walk_secret_ref("${secrets.K}", service_env=service, shared_env=shared)
    assert (res.value, res.source) == (None, None)
    assert res.matched is True
    assert shared.calls == 0


def test_walk_shared_ref_prefers_the_per_service_override():
    service, shared = _Loader({"K": "own"}), _Loader({"K": "global"})
    res = walk_secret_ref("${secrets.shared.K}", service_env=service, shared_env=shared)
    assert (res.value, res.source) == ("own", "service")
    assert shared.calls == 0


def test_walk_shared_ref_falls_through_to_the_shared_scope():
    service, shared = _Loader({}), _Loader({"K": "global"})
    res = walk_secret_ref("${secrets.shared.K}", service_env=service, shared_env=shared)
    assert (res.scope, res.key) == ("shared", "K")
    assert (res.value, res.source) == ("global", "shared")
    assert (service.calls, shared.calls) == (1, 1)


def test_walk_shared_ref_missing_everywhere_is_a_matched_miss():
    res = walk_secret_ref("${secrets.shared.K}", service_env=_Loader({}), shared_env=_Loader({}))
    assert (res.matched, res.value, res.source) == (True, None, None)


@pytest.mark.parametrize("ref", ["", "not-a-ref", "${secrets.k}", "$secrets.K", "${secrets.K}x"])
def test_walk_bad_grammar_never_touches_a_scope(ref):
    service, shared = _Loader({"K": "own"}), _Loader({"K": "global"})
    res = walk_secret_ref(ref, service_env=service, shared_env=shared)
    assert (res.matched, res.scope, res.key, res.value, res.source) == (
        False,
        None,
        None,
        None,
        None,
    )
    assert (service.calls, shared.calls) == (0, 0)


def test_walk_without_a_service_loader_still_reads_the_shared_scope():
    shared = _Loader({"K": "global"})
    res = walk_secret_ref("${secrets.shared.K}", service_env=None, shared_env=shared)
    assert (res.value, res.source) == ("global", "shared")


def test_walk_without_a_service_loader_refuses_an_unscoped_ref():
    # The shared-scope-ONLY posture (WP9 webhook targets, the fresh-deploy
    # carve-out): with no per-service scope to read, an unscoped ref has nowhere
    # left to resolve — it does NOT fall back to a same-named shared key.
    shared = _Loader({"K": "global"})
    res = walk_secret_ref("${secrets.K}", service_env=None, shared_env=shared)
    assert (res.matched, res.value, res.source) == (True, None, None)
    assert shared.calls == 0


def test_walk_without_a_shared_loader_misses_a_shared_ref():
    res = walk_secret_ref("${secrets.shared.K}", service_env=_Loader({}), shared_env=None)
    assert (res.matched, res.value) == (True, None)


def test_walk_propagates_a_loader_failure():
    # Every call site keeps its own handling, so a scope that cannot be read
    # surfaces unchanged rather than being flattened into a plain miss.
    def boom():
        raise SecretDecryptError("scope unreadable")

    with pytest.raises(SecretDecryptError):
        walk_secret_ref("${secrets.K}", service_env=boom, shared_env=_Loader({}))
