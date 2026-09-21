"""`[project]` / `[services.*]` / `[vars]` declaration parsing (P40d, D-P40-12 / D-P40-14)."""

from __future__ import annotations

import tomllib

import pytest

from nerdit.cli.commands.init import NERDIT_TOML_TEMPLATE
from nerdit.config.project import (
    DeclarationError,
    DeployConfig,
    is_project_declaration,
    load_project_config,
    parse_project_declaration,
)

_SECRET = "hunter2-PASTED-CREDENTIAL"


def _decl(**over) -> dict:
    data: dict = {
        "project": {"name": "asso"},
        "services": {"web": {"port": 8000}, "api": {"build_settings": {"subdir": "apps/api"}}},
        "vars": {"required": ["API_KEY", "DB_URL"]},
    }
    data.update(over)
    return {key: value for key, value in data.items() if value is not None}


def _refused(data: dict, code: str = "project.invalid_declaration", **kw) -> str:
    with pytest.raises(DeclarationError) as info:
        parse_project_declaration(data, **kw)
    assert info.value.code == code
    assert _SECRET not in str(info.value)
    return str(info.value)


def test_a_valid_declaration_returns_name_raw_tables_and_required():
    name, services, required = parse_project_declaration(_decl())
    assert name == "asso"
    assert services == {"web": {"port": 8000}, "api": {"build_settings": {"subdir": "apps/api"}}}
    assert list(services) == ["web", "api"]
    assert required == ["API_KEY", "DB_URL"]


def test_vars_is_optional_and_required_defaults_to_empty():
    assert parse_project_declaration(_decl(vars=None))[2] == []
    assert parse_project_declaration(_decl(vars={}))[2] == []


def test_the_returned_tables_are_copies_without_a_name():
    data = _decl()
    _, services, _ = parse_project_declaration(data)
    services["web"]["port"] = 1
    assert data["services"]["web"]["port"] == 8000
    assert "name" not in services["web"]


def test_declaration_error_is_a_value_error():
    assert issubclass(DeclarationError, ValueError)


def test_no_new_model_deploy_config_fields_are_unchanged():
    # The declaration reuses DeployConfig; a new field here is a schema change
    # the shape-hint and MCP docstring pins must learn about first.
    assert "services" not in DeployConfig.model_fields
    assert "project" not in DeployConfig.model_fields


@pytest.mark.parametrize("project", [None, "asso", ["asso"], {}, {"name": 3}])
def test_project_must_be_a_table_with_a_string_name(project):
    code = "project.invalid_name" if isinstance(project, dict) else "project.invalid_declaration"
    _refused(_decl(project=project), code)


def test_project_takes_only_name():
    assert "'name'" in _refused(_decl(project={"name": "asso", "region": _SECRET}))


@pytest.mark.parametrize(
    "name", ["Asso", "my--app", "-asso", "asso-", "a" * 41, "", "as so", _SECRET]
)
def test_new_project_name_grammar(name):
    _refused(_decl(project={"name": name}), "project.invalid_name")


def test_forty_chars_is_the_longest_new_name():
    assert parse_project_declaration(_decl(project={"name": "a" * 40}))[0] == "a" * 40


def test_an_existing_implicit_project_is_grammar_exempt_but_still_a_dns_label():
    data = _decl(project={"name": "my--app"})
    assert parse_project_declaration(data, new_project=False)[0] == "my--app"
    _refused(_decl(project={"name": "My--App"}), "project.invalid_name", new_project=False)
    _refused(_decl(project={"name": "a" * 64}), "project.invalid_name", new_project=False)


def test_the_63_octet_label_refusal():
    # A legacy 60-char project: `web` keeps the bare name, `api--<60>` cannot exist.
    long = "a" * 60
    web_only = _decl(project={"name": long}, services={"web": {}})
    assert parse_project_declaration(web_only, new_project=False)[0] == long
    msg = _refused(_decl(project={"name": long}), "project.label_too_long", new_project=False)
    assert "[services.api]" in msg and long not in msg


@pytest.mark.parametrize("services", [None, {}, "web", ["web"]])
def test_services_must_be_a_non_empty_table(services):
    _refused(_decl(services=services))


@pytest.mark.parametrize("svc", ["Api", "a--b", "-api", "a" * 21, "", _SECRET])
def test_service_name_grammar(svc):
    _refused(_decl(services={svc: {}}), "project.invalid_service")


def test_a_service_must_be_a_table():
    assert "[services.api]" in _refused(_decl(services={"api": _SECRET}))


def test_a_service_table_must_not_name_itself():
    assert "'name'" in _refused(_decl(services={"api": {"name": "other"}}))


def test_a_service_table_must_not_set_auto_deploy():
    assert "'auto_deploy'" in _refused(_decl(services={"api": {"auto_deploy": True}}))


@pytest.mark.parametrize(
    "table",
    [
        {"port": 70000},
        {"gpus": -1},
        {"memory_limit": _SECRET},
        {"release": "a\nb"},
        {"volumes": ["../etc:/data"]},
        {"edge_auth": {"user": "u", "password": _SECRET}},
        {"build_settings": {"subdir": "../up"}},
    ],
)
def test_every_deploy_config_validator_fires_per_service(table):
    assert "[services.api]" in _refused(_decl(services={"web": {}, "api": table}))


def test_the_vars_rewrite_is_accepted_in_a_service_table():
    data = _decl(services={"web": {"edge_auth": {"user": "u", "password": "${vars.PW}"}}})
    assert parse_project_declaration(data)[1]["web"]["edge_auth"]["password"] == "${vars.PW}"


def test_unknown_service_keys_are_tolerated_like_deploy():
    assert "web" in parse_project_declaration(_decl(services={"web": {"future_key": 1}}))[1]


@pytest.mark.parametrize(
    "section",
    ["API_KEY", ["API_KEY"], {"required": "API_KEY"}, {"required": ["API_KEY"], "optional": []}],
)
def test_vars_shape(section):
    _refused(_decl(vars=section))


@pytest.mark.parametrize("key", ["api_key", "1KEY", "KEY-A", "", "${vars.KEY}", _SECRET, 3])
def test_required_keys_follow_the_ref_grammar(key):
    _refused(_decl(vars={"required": ["OK", key]}))


def test_a_trailing_newline_is_not_a_key():
    _refused(_decl(vars={"required": ["KEY\n"]}))


def test_duplicate_required_keys_are_refused():
    assert "more than once" in _refused(_decl(vars={"required": ["A", "B", "A"]}))


def test_deploy_beside_a_declaration_is_refused_value_free():
    msg = _refused(_decl(deploy={"name": _SECRET}))
    assert "[deploy]" in msg and "[project]" in msg


def test_top_level_bindings_go_through_the_shared_schemas():
    ok = _decl(
        ai={"default": {"provider": "ollama", "model": "llama3.1:8b"}},
        db={"default": {"provider": "managed", "database": "pg"}},
    )
    assert parse_project_declaration(ok)[0] == "asso"
    _refused(_decl(ai={"default": {"provider": "api", "model": "m", "api_key": _SECRET}}))
    _refused(_decl(ai="text"))
    _refused(_decl(db={"default": "text"}))


def test_a_declared_engine_is_refused_with_the_intake_hint():
    msg = _refused(_decl(db={"default": {"engine": "postgres"}}))
    assert "nerdit db create" in msg and "database =" in msg
    plain = _refused(_decl(db={"default": {"provider": "managed"}}))
    assert "nerdit db create" not in plain


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"project": {"name": "asso"}}, True),
        ({"services": {}}, True),
        ({"project": "half-written"}, True),
        ({"deploy": {"name": "asso"}}, False),
        ({"ai": {}, "db": {}, "vars": {}}, False),
        ({}, False),
    ],
)
def test_is_project_declaration(data, expected):
    assert is_project_declaration(data) is expected


def test_load_project_config_is_unchanged_for_a_legacy_file(tmp_path):
    path = tmp_path / "nerdit.toml"
    path.write_text('[deploy]\nname = "asso"\nport = 8000\n')
    config = load_project_config(path)
    assert config is not None and config.deploy is not None
    assert (config.deploy.name, config.deploy.port) == ("asso", 8000)


def test_the_init_template_keeps_every_section_commented_and_its_declaration_parses():
    assert tomllib.loads(NERDIT_TOML_TEMPLATE) == {}
    assert "# [deploy]" in NERDIT_TOML_TEMPLATE
    block = NERDIT_TOML_TEMPLATE.split("# [project]")[1].split("# Declare the AI")[0]
    text = "[project]" + "\n".join(line.removeprefix("#").strip() for line in block.splitlines())
    name, services, required = parse_project_declaration(tomllib.loads(text))
    assert (name, list(services), required) == ("my-app", ["web", "api"], ["API_KEY"])
