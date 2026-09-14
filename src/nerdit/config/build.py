"""Bounded build overrides shared by project, API and persisted configuration."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from nerdit.core.node_packages import validate_package_manager
from nerdit.core.node_runtime import SUPPORTED_NODE_VERSIONS

_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")

_PUBLIC_ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_PUBLIC_ENV_MAX_ENTRIES = 64
# Names the buildpack, the package managers or docker itself own; a caller
# override would change how the image builds rather than what the app reads.
# The proxy and BuildKit names are docker's PREDEFINED build args: they take
# effect with no `ARG` declaration at all, in either case, which is why the
# comparison below folds case.
_RESERVED_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "NODE_ENV",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "FTP_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "SOURCE_DATE_EPOCH",
    }
)
_RESERVED_ENV_PREFIXES = ("BUILDKIT_", "COREPACK_", "DOCKER_", "NPM_CONFIG_", "PIP_")

# These persisted commands are not a secret channel. Catch recognizable credential
# syntax; opaque literals and shell obfuscation cannot be identified reliably.
_CREDENTIAL = re.compile(
    r"(?:\b(?:proxy-)?authorization\s*[:=]|\bbearer\s+\S+)"
    r"|(?:\b(?:[a-z_][a-z0-9_]*_)?(?:token|password|passwd|secret|api_key|access_key(?:_id)?|secret_key)\s*=)"
    r"|(?:--(?:token|password|passwd|secret|api[-_]key|access[-_]token|auth[-_]token)(?:=|\s+))"
    r"|(?:\b_authToken\s*(?:=|\s))"
    r"|(?:(?:https?|git\+https?|ftp)://[^/\s\"']+@)"
    r"|(?:[?&](?:token|access_token|api_key|password|secret)=)",
    re.IGNORECASE,
)


class BuildSettings(BaseModel):
    """Null resets detection; false explicitly disables compilation.

    Build environment is supported as ``public_env`` — values land in the build
    output (a browser bundle) and are public by definition. Credentials and
    secret references remain unsupported: commands and public values execute or
    appear only inside the image build, and runtime secrets are never resolved
    into them.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, strict=True)

    preset: Literal["node", "nextjs", "python", "dockerfile"] | None = None
    install: str | None = None
    build: str | Literal[False] | None = None
    start: str | None = None
    node_version: str | None = None
    package_manager: str | None = None
    subdir: str | None = None
    public_env: dict[str, str] | None = None

    @field_validator("install", "build", "start", mode="before")
    @classmethod
    def _command(cls, value: object) -> object:
        if value is None or value is False:
            return value
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 4096
            or _CONTROL.search(value)
        ):
            raise ValueError(
                "Build commands must be non-empty single lines of at most 4096 characters."
            )
        if "${secrets." in value or "${github." in value or _CREDENTIAL.search(value):
            raise ValueError("Build credentials and secret references are unsupported.")
        return value

    @field_validator("build", mode="before")
    @classmethod
    def _strict_false(cls, value: object) -> object:
        if value is not None and not isinstance(value, str) and value is not False:
            raise ValueError("Build must be a command, false, or null.")
        return value

    @field_validator("node_version")
    @classmethod
    def _runtime(cls, value: str | None) -> str | None:
        if value is not None and value not in SUPPORTED_NODE_VERSIONS:
            raise ValueError("Node runtime must be a supported exact version: 22.23.2 or 24.20.0.")
        return value

    @field_validator("package_manager")
    @classmethod
    def _manager(cls, value: str | None) -> str | None:
        if value is not None:
            validate_package_manager(value)
        return value

    @field_validator("subdir")
    @classmethod
    def _root(cls, value: str | None) -> str | None:
        if value is not None and (
            not value
            or len(value) > 512
            or value.startswith("/")
            or "\\" in value
            or ":" in value
            or _CONTROL.search(value)
            or any(part in {"", ".."} for part in value.split("/"))
        ):
            raise ValueError("Build subdir must be a bounded relative path without traversal.")
        return value

    @field_validator("public_env")
    @classmethod
    def _public_env(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        if value is None:
            return value
        if len(value) > _PUBLIC_ENV_MAX_ENTRIES:
            raise ValueError(f"public_env accepts at most {_PUBLIC_ENV_MAX_ENTRIES} entries.")
        for key, item in value.items():
            if not _PUBLIC_ENV_KEY.fullmatch(key):
                raise ValueError(
                    "public_env keys must be environment variable names: a letter or '_' "
                    "followed by letters, digits or '_', 1-128 characters."
                )
            folded = key.upper()
            if folded in _RESERVED_ENV_NAMES or folded.startswith(_RESERVED_ENV_PREFIXES):
                raise ValueError("public_env keys must not override builder-controlled names.")
            if len(item) > 4096 or _CONTROL.search(item):
                raise ValueError(
                    "public_env values must be single lines of at most 4096 characters."
                )
            if "${secrets." in item or "${github." in item:
                raise ValueError(
                    "public_env values are embedded in public build output; secret references "
                    "are not allowed."
                )
        return value
