"""Contract tests for the ``[db.*]`` binding schema (P15 / D-A).

Parallel to ``tests/test_ai_binding_schema.py`` (which stays byte-frozen): the
``[db.*]`` grammar, the provider field rules, the external-URL shape rejects
(scheme / userinfo password / query string), and the round-trip persistence
shape. ``DbBindingConfig`` is a NEW section model — the frozen ``[ai.*]`` block
is never touched.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nerdit.config.project import (
    DB_BINDING_NAME_RE,
    DbBindingConfig,
    ProjectConfig,
    load_project_config,
    parse_db_bindings,
    shared_secret_keys_for,
)

# --- Grammar -------------------------------------------------------------------


def test_db_binding_name_grammar():
    assert DB_BINDING_NAME_RE.pattern == r"^[a-z][a-z0-9_]{0,31}$"


@pytest.mark.parametrize(
    "bad_name",
    ["Default", "1abc", "_x", "a-b", "", "a" * 33],
)
def test_bad_binding_name_rejected(bad_name):
    binding = DbBindingConfig(provider="managed", database="pg")
    with pytest.raises(ValidationError, match="binding name"):
        ProjectConfig(db={bad_name: binding})


@pytest.mark.parametrize("good_name", ["default", "a", "cache_2", "a" * 32])
def test_good_binding_name_accepted(good_name):
    binding = DbBindingConfig(provider="managed", database="pg")
    config = ProjectConfig(db={good_name: binding})
    assert config.db is not None
    assert good_name in config.db


# --- Provider field rules ------------------------------------------------------


def test_managed_requires_database():
    with pytest.raises(ValidationError, match="requires 'database'"):
        DbBindingConfig(provider="managed")


def test_managed_forbids_url():
    with pytest.raises(ValidationError, match="forbids 'url'"):
        DbBindingConfig(provider="managed", database="pg", url="postgresql://host/db")


def test_managed_forbids_password():
    with pytest.raises(ValidationError, match="forbids 'password'"):
        DbBindingConfig(provider="managed", database="pg", password="${secrets.PW}")


def test_managed_minimal_accepted():
    cfg = DbBindingConfig(provider="managed", database="pg")
    assert cfg.provider == "managed"
    assert cfg.database == "pg"
    assert cfg.url is None
    assert cfg.password is None


def test_external_requires_url():
    with pytest.raises(ValidationError, match="requires 'url'"):
        DbBindingConfig(provider="external", password="${secrets.PW}")


def test_external_requires_password():
    with pytest.raises(ValidationError, match="requires 'password'"):
        DbBindingConfig(provider="external", url="postgresql://user@host:5432/db")


def test_external_forbids_database():
    with pytest.raises(ValidationError, match="forbids 'database'"):
        DbBindingConfig(
            provider="external",
            database="pg",
            url="postgresql://user@host:5432/db",
            password="${secrets.PW}",
        )


def test_external_accepted():
    cfg = DbBindingConfig(
        provider="external",
        url="postgresql://user@db.example.com:5432/app",
        password="${secrets.DB_PASSWORD}",
    )
    assert cfg.url == "postgresql://user@db.example.com:5432/app"
    assert cfg.password == "${secrets.DB_PASSWORD}"


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        DbBindingConfig(provider="mongo", url="mongodb://host")


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        DbBindingConfig(provider="managed", database="pg", databse="typo")


# --- External URL shape --------------------------------------------------------


@pytest.mark.parametrize(
    "scheme_url",
    [
        "postgresql://user@host:5432/db",
        "redis://host:6379/0",
        "rediss://host:6379/0",
    ],
)
def test_good_url_schemes_accepted(scheme_url):
    cfg = DbBindingConfig(provider="external", url=scheme_url, password="${secrets.PW}")
    assert cfg.url == scheme_url


@pytest.mark.parametrize(
    "bad_url",
    [
        "mysql://host/db",  # unsupported scheme
        "postgres://host/db",  # not the postgresql:// form
        "postgresql+asyncpg://host/db",  # driver-qualified scheme
        "host:5432/db",  # no scheme
    ],
)
def test_bad_scheme_rejected(bad_url):
    with pytest.raises(ValidationError, match="scheme must be one of"):
        DbBindingConfig(provider="external", url=bad_url, password="${secrets.PW}")


@pytest.mark.parametrize(
    "userinfo_url",
    [
        "postgresql://user:secret@host:5432/db",
        "redis://:pw@host:6379/0",
    ],
)
def test_userinfo_password_rejected(userinfo_url):
    with pytest.raises(ValidationError, match="must not embed a password"):
        DbBindingConfig(provider="external", url=userinfo_url, password="${secrets.PW}")


@pytest.mark.parametrize(
    "query_url",
    [
        "postgresql://user@host:5432/db?sslmode=require",
        "postgresql://user@host:5432/db?password=x",
        "redis://host:6379/0?foo=bar",
    ],
)
def test_query_string_rejected(query_url):
    with pytest.raises(ValidationError, match="query string is not allowed"):
        DbBindingConfig(provider="external", url=query_url, password="${secrets.PW}")


# --- password: the ${secrets.KEY} grammar --------------------------------------


@pytest.mark.parametrize(
    "bad_pw",
    [
        "hunter2",  # literal
        "secrets.PW",  # missing ${...}
        "${secrets.lower}",  # lowercase key
        "${secret.PW}",  # wrong namespace
        "${secrets.other.PW}",  # unknown scope
    ],
)
def test_literal_or_malformed_password_rejected(bad_pw):
    with pytest.raises(ValidationError, match="secret reference"):
        DbBindingConfig(provider="external", url="postgresql://user@host/db", password=bad_pw)


@pytest.mark.parametrize(
    "good_pw",
    ["${secrets.PW}", "${secrets.DB_PASSWORD}", "${secrets.shared.PW}"],
)
def test_wellformed_password_accepted(good_pw):
    cfg = DbBindingConfig(provider="external", url="postgresql://user@host/db", password=good_pw)
    assert cfg.password == good_pw


# --- [db] table-of-tables shape ------------------------------------------------


def test_db_section_must_be_table():
    with pytest.raises(ValueError, match=r"\[db\] must be a table"):
        parse_db_bindings("managed")


def test_db_binding_must_be_table(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[db]\ndefault = 'managed'\n")
    with pytest.raises(ValueError, match=r"\[db\.default\] must be a table"):
        load_project_config(path=toml_file)


# --- Round-trip ----------------------------------------------------------------


def test_north_star_toml_parses(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text(
        "[deploy]\nname = 'my-app'\n\n[db.default]\nprovider = 'managed'\ndatabase = 'pg'\n"
    )
    config = load_project_config(path=toml_file)
    assert config is not None and config.db is not None
    assert config.db["default"].provider == "managed"
    assert config.db["default"].database == "pg"


def test_db_absent_yields_none(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text('[run]\nscript = "train.py"\n')
    config = load_project_config(path=toml_file)
    assert config is not None
    assert config.db is None


def test_db_bindings_round_trip_model_dump():
    config = ProjectConfig(
        db={
            "default": DbBindingConfig(provider="managed", database="pg"),
            "cache": DbBindingConfig(
                provider="external",
                url="redis://host:6379/0",
                password="${secrets.shared.REDIS_PW}",
            ),
        }
    )
    dumped = config.model_dump()
    restored = ProjectConfig(**dumped)
    assert restored == config


# --- shared_secret_keys_for (P15 generalized scanner) --------------------------


def test_shared_secret_keys_for_password_field():
    specs = {
        "managed": {"provider": "managed", "database": "pg"},
        "ext": {
            "provider": "external",
            "url": "postgresql://u@h/db",
            "password": "${secrets.PW}",  # unscoped — not shared
        },
        "shared_ext": {
            "provider": "external",
            "url": "redis://h/0",
            "password": "${secrets.shared.REDIS_PW}",
        },
    }
    assert shared_secret_keys_for(specs, ("password",)) == ["REDIS_PW"]


def test_shared_secret_keys_for_tolerates_malformed():
    assert shared_secret_keys_for(None, ("password",)) == []
    assert shared_secret_keys_for({"bad": "x", "worse": {"password": 42}}, ("password",)) == []
