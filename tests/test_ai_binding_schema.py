"""Contract-freeze tests for the ``[ai.*]`` binding schema (P5 / S1).

These tests pin the **frozen** ``[ai.*]`` grammar (Invariant #1): binding
names, the ``${secrets.KEY}`` api_key reference, and the per-provider field
rules. Loosening any of them is a contract change and needs a review — not a
quiet edit here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nerdit.config.project import (
    AI_BINDING_NAME_RE,
    SECRET_REF_RE,
    AiBindingConfig,
    DbBindingConfig,
    DeployConfig,
    EdgeAuthConfig,
    ProjectConfig,
    load_project_config,
    parse_ai_bindings,
    rewrite_vars_ref,
    shared_secret_keys,
)

# --- Frozen grammar (literal pins) ---------------------------------------------


def test_binding_name_grammar_is_frozen():
    assert AI_BINDING_NAME_RE.pattern == r"^[a-z][a-z0-9_]{0,31}$"


def test_secret_ref_grammar_is_frozen():
    # P8 deliberately extended the P5 grammar with the optional 'shared' scope
    # (contract-freeze review). Two groups: (scope, KEY);
    # legacy unscoped refs parse unchanged with scope=None.
    assert SECRET_REF_RE.pattern == r"^\$\{secrets\.(?:(shared)\.)?([A-Z][A-Z0-9_]*)\}$"


def test_secret_ref_groups_are_scope_then_key():
    unscoped = SECRET_REF_RE.match("${secrets.OPENAI_KEY}")
    assert unscoped is not None
    assert unscoped.groups() == (None, "OPENAI_KEY")
    scoped = SECRET_REF_RE.match("${secrets.shared.OPENAI_KEY}")
    assert scoped is not None
    assert scoped.groups() == ("shared", "OPENAI_KEY")


# --- (P40c / D-P40-9) the `${vars.…}` alias is rewritten at ingress -------------


@pytest.mark.parametrize(
    ("alias", "stored"),
    [
        ("${vars.OPENAI_KEY}", "${secrets.OPENAI_KEY}"),
        ("${vars.shared.OPENAI_KEY}", "${secrets.shared.OPENAI_KEY}"),
    ],
)
def test_vars_alias_is_accepted_and_stored_as_the_secrets_form(alias, stored):
    ai = AiBindingConfig(provider="api", model="m", base_url="https://x.example/v1", api_key=alias)
    assert ai.api_key == stored
    assert ai.model_dump(exclude_none=True)["api_key"] == stored
    db = DbBindingConfig(provider="external", url="postgresql://db.example/app", password=alias)
    assert db.password == stored
    edge = EdgeAuthConfig(user="ops", password=alias)
    assert edge.password == stored
    assert rewrite_vars_ref(alias) == stored


def test_the_alias_rewrite_leaves_everything_else_to_the_frozen_grammar():
    assert rewrite_vars_ref(None) is None
    assert rewrite_vars_ref("${secrets.K}") == "${secrets.K}"
    assert rewrite_vars_ref("sk-literal") == "sk-literal"
    for bad in ("${vars.lower}", "${vars.other.KEY}", "x${vars.KEY}", "${vars.}"):
        with pytest.raises(ValidationError) as excinfo:
            AiBindingConfig(provider="api", model="m", base_url="https://x.example/v1", api_key=bad)
        assert "api_key" in str(excinfo.value)
        assert bad not in str(excinfo.value)


# --- The north-star two-binding example parses ---------------------------------


def test_two_binding_example_parses(tmp_path):
    """The plan's demo nerdit.toml — one local ollama + one external api binding."""
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text(
        "[deploy]\n"
        "name = 'my-app'\n"
        "port = 8000\n"
        "\n"
        "[ai.default]\n"
        "provider = 'ollama'\n"
        "model = 'llama3.1:8b'\n"
        "\n"
        "[ai.cheap]\n"
        "provider = 'api'\n"
        "model = 'gpt-4o-mini'\n"
        "base_url = 'https://api.openai.com/v1'\n"
        'api_key = "${secrets.OPENAI_KEY}"\n'
    )
    config = load_project_config(path=toml_file)
    assert config is not None
    assert config.ai is not None
    assert set(config.ai) == {"default", "cheap"}

    default = config.ai["default"]
    assert default.provider == "ollama"
    assert default.model == "llama3.1:8b"
    assert default.base_url is None
    assert default.api_key is None

    cheap = config.ai["cheap"]
    assert cheap.provider == "api"
    assert cheap.model == "gpt-4o-mini"
    assert cheap.base_url == "https://api.openai.com/v1"
    assert cheap.api_key == "${secrets.OPENAI_KEY}"


def test_ai_absent_yields_none(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text('[run]\nscript = "train.py"\n')
    config = load_project_config(path=toml_file)
    assert config is not None
    assert config.ai is None


# --- Provider field rules -------------------------------------------------------


def test_api_without_base_url_rejected():
    with pytest.raises(ValidationError, match="requires 'base_url'"):
        AiBindingConfig(provider="api", model="gpt-4o-mini", api_key="${secrets.OPENAI_KEY}")


def test_api_without_api_key_rejected():
    """provider='api' requires api_key: S9 resolution always resolves a secret ref."""
    with pytest.raises(ValidationError, match="requires 'api_key'"):
        AiBindingConfig(provider="api", model="gpt-4o-mini", base_url="https://api.openai.com/v1")


def test_ollama_with_base_url_rejected():
    with pytest.raises(ValidationError, match="forbids 'base_url'"):
        AiBindingConfig(
            provider="ollama", model="llama3.1:8b", base_url="http://127.0.0.1:11434/v1"
        )


def test_ollama_with_api_key_rejected():
    """The daemon injects the local placeholder key — user keys are meaningless."""
    with pytest.raises(ValidationError, match="forbids 'api_key'"):
        AiBindingConfig(provider="ollama", model="llama3.1:8b", api_key="${secrets.OPENAI_KEY}")


def test_ollama_with_shared_api_key_rejected():
    """The P8 shared scope changes nothing here: ollama still forbids api_key."""
    with pytest.raises(ValidationError, match="forbids 'api_key'"):
        AiBindingConfig(
            provider="ollama", model="llama3.1:8b", api_key="${secrets.shared.OPENAI_KEY}"
        )


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        AiBindingConfig(provider="vllm", model="llama3.1:8b")


def test_missing_model_rejected():
    with pytest.raises(ValidationError):
        AiBindingConfig(provider="ollama")


def test_empty_model_rejected():
    with pytest.raises(ValidationError):
        AiBindingConfig(provider="ollama", model="")


def test_unknown_key_rejected_extra_forbid():
    """Typos fail loudly instead of being silently ignored."""
    with pytest.raises(ValidationError):
        AiBindingConfig(provider="ollama", model="llama3.1:8b", modle="oops")


# --- api_key: the frozen ${secrets.KEY} grammar ---------------------------------


@pytest.mark.parametrize(
    "bad_key",
    [
        "sk-abc123",  # literal API key
        "secrets.OPENAI_KEY",  # missing ${...}
        "${secret.OPENAI_KEY}",  # wrong namespace
        "${secrets.openai_key}",  # lowercase key
        "${secrets.1KEY}",  # must start with a letter
        "${secrets.}",  # empty key
        " ${secrets.OPENAI_KEY}",  # leading whitespace
        "${secrets.OPENAI_KEY} ",  # trailing whitespace
        "${secrets.OPENAI-KEY}",  # '-' not in the grammar
        "${secrets.other.KEY}",  # 'shared' is the only scope (P8)
        "${secrets.Shared.KEY}",  # scope is the literal lowercase 'shared'
        "${secrets.shared.lower}",  # scoped key still uppercase
        "${secrets.shared.}",  # scoped empty key
        "${secrets.shared.A.B}",  # no nested scopes
    ],
)
def test_literal_or_malformed_api_key_rejected(bad_key):
    with pytest.raises(ValidationError, match="secret reference"):
        AiBindingConfig(
            provider="api",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=bad_key,
        )


@pytest.mark.parametrize(
    "good_key",
    [
        "${secrets.K}",
        "${secrets.OPENAI_KEY_2}",
        "${secrets.shared.K}",  # shared scope (P8)
        "${secrets.shared.OPENAI_KEY}",
    ],
)
def test_wellformed_api_key_accepted(good_key):
    cfg = AiBindingConfig(
        provider="api",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=good_key,
    )
    assert cfg.api_key == good_key


# --- Binding names --------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "Default",  # uppercase
        "1abc",  # must start with a letter
        "_x",  # must start with a letter
        "a-b",  # '-' not allowed (env-var mapping)
        "",  # empty
        "a" * 33,  # too long (max 32)
    ],
)
def test_bad_binding_name_rejected(bad_name):
    binding = AiBindingConfig(provider="ollama", model="llama3.1:8b")
    with pytest.raises(ValidationError, match="binding name"):
        ProjectConfig(ai={bad_name: binding})


@pytest.mark.parametrize("good_name", ["default", "a", "cheap_2", "a" * 32])
def test_good_binding_name_accepted(good_name):
    binding = AiBindingConfig(provider="ollama", model="llama3.1:8b")
    config = ProjectConfig(ai={good_name: binding})
    assert config.ai is not None
    assert good_name in config.ai


def test_bad_binding_name_rejected_from_toml(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[ai.BadName]\nprovider = 'ollama'\nmodel = 'llama3.1:8b'\n")
    with pytest.raises(ValidationError, match="binding name"):
        load_project_config(path=toml_file)


# --- [ai] table-of-tables shape -------------------------------------------------


def test_ai_section_must_be_table():
    with pytest.raises(ValueError, match=r"\[ai\] must be a table"):
        parse_ai_bindings("ollama")


def test_ai_binding_must_be_table(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[ai]\ndefault = 'ollama'\n")
    with pytest.raises(ValueError, match=r"\[ai\.default\] must be a table"):
        load_project_config(path=toml_file)


def test_ai_binding_typo_key_rejected_from_toml(tmp_path):
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text("[ai.default]\nprovider = 'ollama'\nmodel = 'llama3.1:8b'\nmodle = 'x'\n")
    with pytest.raises(ValidationError):
        load_project_config(path=toml_file)


# --- Round-trip -----------------------------------------------------------------


def test_ai_bindings_round_trip_model_dump():
    """The spec dict persisted in jobs.config must re-validate unchanged (S8)."""
    config = ProjectConfig(
        ai={
            "default": AiBindingConfig(provider="ollama", model="llama3.1:8b"),
            "cheap": AiBindingConfig(
                provider="api",
                model="gpt-4o-mini",
                base_url="https://api.openai.com/v1",
                api_key="${secrets.OPENAI_KEY}",
            ),
        }
    )
    dumped = config.model_dump()
    assert dumped["ai"] == {
        "default": {
            "provider": "ollama",
            "model": "llama3.1:8b",
            "base_url": None,
            "api_key": None,
        },
        "cheap": {
            "provider": "api",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key": "${secrets.OPENAI_KEY}",
        },
    }
    restored = ProjectConfig(**dumped)
    assert restored == config


def test_shared_ref_round_trips_through_toml(tmp_path):
    """A ${secrets.shared.KEY} ref parses from nerdit.toml and dumps unchanged (P8)."""
    toml_file = tmp_path / "nerdit.toml"
    toml_file.write_text(
        "[ai.cheap]\n"
        'provider = "api"\n'
        'model = "gpt-4o-mini"\n'
        'base_url = "https://api.openai.com/v1"\n'
        'api_key = "${secrets.shared.OPENAI_KEY}"\n'
    )
    config = load_project_config(path=toml_file)
    assert config is not None and config.ai is not None
    assert config.ai["cheap"].api_key == "${secrets.shared.OPENAI_KEY}"
    dumped = config.model_dump()["ai"]["cheap"]
    assert dumped["api_key"] == "${secrets.shared.OPENAI_KEY}"


# --- shared_secret_keys (P8) -----------------------------------------------------


def test_shared_secret_keys_extracts_only_shared_scoped_refs():
    specs = {
        "default": {"provider": "ollama", "model": "llama3.1:8b"},
        "cheap": {
            "provider": "api",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "api_key": "${secrets.OPENAI_KEY}",  # unscoped — not shared
        },
        "smart": {
            "provider": "api",
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "${secrets.shared.OPENAI_KEY}",
        },
        "other": {
            "provider": "api",
            "model": "m",
            "base_url": "https://x/v1",
            "api_key": "${secrets.shared.ANTHROPIC_KEY}",
        },
    }
    assert shared_secret_keys(specs) == ["ANTHROPIC_KEY", "OPENAI_KEY"]


def test_shared_secret_keys_tolerates_absent_and_malformed_specs():
    assert shared_secret_keys(None) == []
    assert shared_secret_keys({}) == []
    # A loosely-typed persisted blob never raises — it just contributes nothing.
    assert shared_secret_keys({"bad": "not-a-table", "worse": {"api_key": 42}}) == []


# --- Bug #28 / P25 WP0: the rejected api_key never rides the error string -------

# The canary the §5.1 suite hunts across every sink (this file pins the model
# layer; tests/test_ai_key_echo_leak.py pins the routes). Deliberately shaped
# like a real provider key so a partial redaction still reads as a failure.
CANARY = "sk-live-CANARY-0000"


def test_rejected_api_key_is_absent_from_the_validation_error():
    """P25 WP0 (bug #28): neither pydantic nor the validator echoes the value.

    Two load-bearing assertions, one per fix — the mutation check the plan §5.1
    demands. Reverting ``hide_input_in_errors=True`` re-adds pydantic's
    ``input_value=`` tail (assertion 1); restoring the validator's ``'{value}'``
    interpolation re-adds the value to ``msg`` (assertion 2). Either revert alone
    turns this test red.
    """
    with pytest.raises(ValidationError) as excinfo:
        AiBindingConfig(
            provider="api",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=CANARY,
        )
    text = str(excinfo.value)
    assert CANARY not in text

    # 1) hide_input_in_errors: pydantic's own "[type=…, input_value='sk-…']" tail.
    assert "input_value=" not in text

    # 2) the validator's own message — the channel hide_input_in_errors never
    #    touches, and the one that rides `errors()[0]['msg']` into deploy.invalid.
    errors = excinfo.value.errors()
    assert CANARY not in errors[0]["msg"]
    assert errors[0]["msg"].startswith("Value error, Invalid [ai] api_key:")


def test_rejected_api_key_keeps_its_diagnostic_location():
    """Positive control: the field name survives so the error stays actionable."""
    with pytest.raises(ValidationError) as excinfo:
        AiBindingConfig(
            provider="api",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key=CANARY,
        )
    assert excinfo.value.errors()[0]["loc"] == ("api_key",)
    assert "api_key" in str(excinfo.value)


def test_parse_ai_bindings_propagates_the_value_free_error():
    """The §3 WP0.3 neighbour verdict, pinned.

    ``parse_ai_bindings`` lets ``AiBindingConfig``'s ValidationError propagate
    verbatim (so the WP0 fixes are the only thing standing between a literal key
    and the 422), and its own shape-error branch interpolates the user-chosen
    binding NAME + a type name — a label, never a credential. Left alone.
    """
    with pytest.raises(ValidationError) as excinfo:
        parse_ai_bindings(
            {
                "cheap": {
                    "provider": "api",
                    "model": "gpt-4o-mini",
                    "base_url": "https://api.openai.com/v1",
                    "api_key": CANARY,
                }
            }
        )
    assert CANARY not in str(excinfo.value)

    with pytest.raises(ValueError, match=r"\[ai\.cheap\] must be a table"):
        parse_ai_bindings({"cheap": "not-a-table"})


# --- P25 WP4a: [deploy].edge_auth, the third secret-ref schema -----------------

# A literal password shaped like a real one, hunted across the error channel the
# same way CANARY is above — this schema is nested in DeployConfig, whose
# ValidationError feeds the ``deploy.invalid`` 422 envelope.
PW_CANARY = "hunter2-CANARY-0000"


def test_edge_auth_accepts_a_secret_ref():
    cfg = EdgeAuthConfig(user="alice", password="${secrets.APP_PW}")
    assert cfg.user == "alice"
    assert cfg.password == "${secrets.APP_PW}"


def test_edge_auth_accepts_a_shared_secret_ref():
    cfg = EdgeAuthConfig(user="alice", password="${secrets.shared.APP_PW}")
    assert cfg.password == "${secrets.shared.APP_PW}"


def test_edge_auth_trailing_newline_password_rejected_at_ingress():
    """A '$'-anchored ``.match`` accepts ``"${secrets.K}\\n"`` — the ingress
    must refuse what ``load_edge_auth``'s fullmatch backstop would refuse, or
    the API returns 201 for a declaration the route plane then withholds as
    malformed on every tick (review-upheld P25 finding)."""
    with pytest.raises(ValidationError) as excinfo:
        EdgeAuthConfig(user="alice", password="${secrets.APP_PW}\n")
    assert excinfo.value.errors()[0]["loc"] == ("password",)


def test_edge_auth_literal_password_is_absent_from_the_validation_error():
    """P25 D-P25-5: the rejected password never rides the error string.

    Two load-bearing assertions, one per protection — the §5.1 mutation idiom
    applied to the new schema. Dropping ``hide_input_in_errors=True`` re-adds
    pydantic's ``input_value=`` tail (assertion 1); interpolating the value into
    the validator's own message re-adds it to ``msg``, the field the
    ``deploy.invalid`` envelope actually surfaces (assertion 2).
    """
    with pytest.raises(ValidationError) as excinfo:
        EdgeAuthConfig(user="alice", password=PW_CANARY)
    text = str(excinfo.value)
    assert PW_CANARY not in text

    # 1) hide_input_in_errors: pydantic's own "[type=…, input_value='hunter2…']".
    assert "input_value=" not in text

    # 2) the validator's own message — the one that rides errors()[0]['msg'].
    errors = excinfo.value.errors()
    assert PW_CANARY not in errors[0]["msg"]
    assert errors[0]["msg"].startswith("Value error, Invalid [deploy] edge_auth password:")


def test_edge_auth_literal_password_keeps_its_diagnostic_location():
    """Positive control: the field name survives so the error stays actionable."""
    with pytest.raises(ValidationError) as excinfo:
        EdgeAuthConfig(user="alice", password=PW_CANARY)
    assert excinfo.value.errors()[0]["loc"] == ("password",)
    assert "password" in str(excinfo.value)


def test_edge_auth_literal_password_stays_masked_through_deploy_config():
    """The nesting that makes this matter: DeployConfig is the 422's source."""
    with pytest.raises(ValidationError) as excinfo:
        DeployConfig(name="app", edge_auth={"user": "alice", "password": PW_CANARY})
    assert PW_CANARY not in str(excinfo.value)
    assert "input_value=" not in str(excinfo.value)
    assert PW_CANARY not in excinfo.value.errors()[0]["msg"]


@pytest.mark.parametrize(
    "bad_user",
    [
        "al:ice",  # RFC 7617 separator
        "",  # min_length
        "   ",  # all-whitespace reads as blank, not as a credential
        "a" * 65,  # max_length
        "ali\nce",  # control character (log forging)
        "ali\x00ce",
        "alicé",  # non-ASCII
        # Trailing newline: '$'-anchored .match accepts it (the one control
        # char the anchor admits) — pins the review-upheld fullmatch at ingress.
        "alice\n",
    ],
)
def test_edge_auth_bad_user_rejected(bad_user):
    with pytest.raises(ValidationError) as excinfo:
        EdgeAuthConfig(user=bad_user, password="${secrets.APP_PW}")
    assert excinfo.value.errors()[0]["loc"] == ("user",)
    # Value-free even for the user: a control character echoed into the 422 /
    # job_logs / diagnose is the log-forging vector the grammar exists to stop.
    assert "input_value=" not in str(excinfo.value)


@pytest.mark.parametrize("good_user", ["alice", "a", "a" * 64, "ops-team_1", "user@example.com"])
def test_edge_auth_good_user_accepted(good_user):
    assert EdgeAuthConfig(user=good_user, password="${secrets.APP_PW}").user == good_user


def test_edge_auth_unknown_key_rejected_at_ingress():
    """``extra='forbid'``: a typo fails loudly instead of silently unprotecting."""
    with pytest.raises(ValidationError) as excinfo:
        EdgeAuthConfig(user="alice", password="${secrets.APP_PW}", passwd=PW_CANARY)
    kinds = {err["type"] for err in excinfo.value.errors()}
    assert "extra_forbidden" in kinds
    assert PW_CANARY not in str(excinfo.value)


def test_edge_auth_missing_password_rejected():
    with pytest.raises(ValidationError) as excinfo:
        EdgeAuthConfig(user="alice")
    assert excinfo.value.errors()[0]["loc"] == ("password",)


def test_deploy_config_edge_auth_defaults_to_none_and_round_trips():
    assert DeployConfig(name="app").edge_auth is None
    cfg = DeployConfig(name="app", edge_auth={"user": "alice", "password": "${secrets.APP_PW}"})
    assert cfg.edge_auth is not None
    # The persisted shape (build_fields writes model_dump() into jobs.config).
    assert cfg.edge_auth.model_dump() == {"user": "alice", "password": "${secrets.APP_PW}"}


def test_edge_auth_parses_from_toml(tmp_path):
    (tmp_path / "nerdit.toml").write_text(
        "[deploy]\n"
        "name = 'my-app'\n"
        "port = 8000\n"
        "\n"
        "[deploy.edge_auth]\n"
        "user = 'alice'\n"
        "password = '${secrets.APP_PW}'\n"
    )
    config = load_project_config(path=tmp_path / "nerdit.toml")
    assert config is not None
    assert config.deploy is not None
    assert config.deploy.edge_auth is not None
    assert config.deploy.edge_auth.user == "alice"


def test_edge_auth_is_a_known_deploy_key(caplog):
    """P25 rev-2 amendment 9: ``_warn_unknown_deploy_keys`` is model_fields-driven.

    There is no known-key set to extend — declaring the field on
    :class:`DeployConfig` is the whole fix, so this pins it rather than the
    (nonexistent) set.
    """
    from nerdit.daemon.deploy_pipeline import _warn_unknown_deploy_keys

    assert "edge_auth" in DeployConfig.model_fields
    with caplog.at_level("WARNING"):
        assert _warn_unknown_deploy_keys({"edge_auth": {"user": "a"}, "name": "app"}, "app") == []
    assert "edge_auth" not in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        # (Agent-DX) the same names now also RETURN, for the deploy-result hint.
        assert _warn_unknown_deploy_keys({"edgeauth": {}}, "app") == ["edgeauth"]
    assert "edgeauth" in caplog.text
