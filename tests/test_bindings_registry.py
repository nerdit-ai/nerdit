"""The per-kind binding registry (P15 / D-A) — `core/bindings/registry.py`.

Pins that :data:`BINDING_KINDS` is the two-field table the diagnose path
consults: every kind is present, and each ``key_names_fn`` round-trips against
its real injector (F7-ENVGRAMMAR — the grammar lives in the injector, the
registry only points at it).
"""

from __future__ import annotations

from nerdit.core.bindings.registry import BINDING_KINDS
from nerdit.core.data.binding import inject_db_env_key_names
from nerdit.core.models.binding import inject_env_key_names


def test_registry_has_both_kinds():
    assert set(BINDING_KINDS) == {"ai", "db"}


def test_ai_entry_points_at_the_frozen_callables():
    ai = BINDING_KINDS["ai"]
    assert ai.key_names_fn is inject_env_key_names
    assert ai.shared_secret_fields == ("api_key",)


def test_db_entry_points_at_the_data_callables():
    db = BINDING_KINDS["db"]
    assert db.key_names_fn is inject_db_env_key_names
    assert db.shared_secret_fields == ("password",)


def test_ai_key_names_fn_round_trips_against_the_real_injector():
    specs = {
        "default": {
            "provider": "api",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key": "${secrets.OPENAI_KEY}",
        },
        "cheap": {
            "provider": "ollama",
            "model": "llama3.1:8b",
        },
    }
    names = BINDING_KINDS["ai"].key_names_fn(specs)
    assert names == inject_env_key_names(specs)
    assert "OPENAI_BASE_URL" in names
    assert "NERDIT_AI_CHEAP_URL" in names


def test_db_key_names_fn_round_trips_against_the_real_injector():
    specs = {
        "default": {"provider": "managed", "database": "pg"},
        "cache": {"provider": "managed", "database": "redis"},
    }
    names = BINDING_KINDS["db"].key_names_fn(specs)
    assert names == inject_db_env_key_names(specs)
    # default binding → the DATABASE_URL alias + every binding → NERDIT_DB_<NAME>_URL.
    assert "DATABASE_URL" in names
    assert "NERDIT_DB_DEFAULT_URL" in names
    assert "NERDIT_DB_CACHE_URL" in names


def test_ai_and_db_injected_key_sets_are_disjoint():
    # Collision-freedom: no cross-kind env name is produced for any name set
    # (the aliases DATABASE_URL/REDIS_URL/OPENAI_* live in different namespaces).
    ai_specs = {"default": {"provider": "ollama", "model": "llama3.1:8b"}}
    db_specs = {"default": {"provider": "managed", "database": "pg"}}
    ai_names = BINDING_KINDS["ai"].key_names_fn(ai_specs)
    db_names = BINDING_KINDS["db"].key_names_fn(db_specs)
    assert ai_names.isdisjoint(db_names)
