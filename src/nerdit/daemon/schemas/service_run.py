"""Schemas for bounded one-off commands in service containers.

Caps bound argv, env and output in requests, last_run storage and audit params.
The route also enforces the configured timeout and log-tail cap; the controller
bounds output capture by bytes.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from nerdit.daemon.schemas._base import StrictRequestModel

# A POSIX-shaped environment variable name. Deliberately stricter than the
# kernel (which accepts any byte string free of '=' and NUL) because this is the
# grammar the `KEY=VALUE` wire format between the daemon and the container
# runtime can round-trip without ambiguity.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

# Per-argument and per-env-value character caps. Both are request bounds only —
# the totals that matter (256 args, 64 keys) are the Field cardinality caps.
_MAX_ARG_CHARS = 4096
_MAX_ENV_VALUE_CHARS = 4096


class ServiceRunRequest(StrictRequestModel):
    """Run a bounded command in a service's current image.

    Command is unshelled: no globbing or pipes. It replaces the image CMD and is
    appended to ENTRYPOINT when present, so it must be valid for that entrypoint.
    The container has no row, ports or GPUs; the request returns HTTP 200.
    Resolved service environment is combined with the supplied env overlay.

    Never put credentials in command: host docker inspect and owner-visible
    last_run expose argv. Use env; its values are masked in audits, scrubbed from
    log tails and excluded from all response/error echoes. Validation errors retain
    the rejected key name but mask secret-bearing input through
    errors._SECRET_INPUT_FIELDS.
    """

    command: list[str] = Field(
        min_length=1,
        max_length=256,
        description=(
            "Argv run unshelled (no shell, so no globbing or pipes): it "
            "replaces the image's CMD and is APPENDED to its ENTRYPOINT if one "
            "is declared. Never put credentials here — argv is host-visible "
            "and stored in last_run"
        ),
    )
    env: dict[str, str] | None = Field(
        default=None,
        max_length=64,
        description=(
            "Env overrides applied to this run only (values masked in audit, "
            "scrubbed from the returned log tail, and never echoed back in any "
            "response — a 422 reports the rejected KEY, never the values). "
            "Platform-injected [ai.*]/[db.*] keys plus "
            "PORT and NERDIT_RUN_ID win over an override; everything else, "
            "config env and secrets included, loses to it"
        ),
    )
    timeout_s: int = Field(
        default=300,
        ge=1,
        description=(
            "Hard server-side cap in seconds; on expiry the container is killed "
            "and the response carries timed_out=true. The route rejects a value "
            "above [services].run_timeout_max_s with 422 run.timeout_too_large"
        ),
    )
    log_tail: int = Field(
        default=200,
        ge=1,
        le=200,
        description=(
            "Lines of scrubbed output to return (le mirrors the route's "
            "_MAX_RUN_LOG_TAIL, which clamps again server-side)"
        ),
    )

    @field_validator("command")
    @classmethod
    def _check_command_args(cls, value: list[str]) -> list[str]:
        """Reject empty, NUL-bearing and oversized arguments.

        NUL cannot survive execve and would make the NUL-joined audit digest ambiguous.
        Other C0 controls, including newlines, are valid argument content. Errors name
        the position, not the value; command remains visible in validation input echoes
        because credentials must be passed through env, never argv.
        """
        for index, arg in enumerate(value):
            if not arg:
                raise ValueError(f"Invalid command: argument {index} is empty.")
            if "\0" in arg:
                raise ValueError(f"Invalid command: argument {index} contains a NUL byte.")
            if len(arg) > _MAX_ARG_CHARS:
                raise ValueError(
                    f"Invalid command: argument {index} exceeds {_MAX_ARG_CHARS} characters."
                )
        return value

    @field_validator("env")
    @classmethod
    def _check_env(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        """Validate complete env key names before KEY=VALUE serialization.

        Equals signs and newlines could bypass protected-name checks. Use fullmatch;
        match with a dollar anchor would accept a trailing newline.
        """
        if value is None:
            return value
        for key, item in value.items():
            if not _ENV_KEY_RE.fullmatch(key):
                # `!r` over a bounded slice: the key is caller-controlled and
                # of unbounded length until this very check, and repr is what
                # renders an embedded newline or NUL harmlessly in the envelope.
                raise ValueError(
                    f"Invalid env key {key[:64]!r}: expected an environment "
                    "variable name (letter or '_' followed by letters, digits "
                    "or '_', 1-128 chars)."
                )
            if len(item) > _MAX_ENV_VALUE_CHARS:
                raise ValueError(
                    f"Invalid env value for '{key}': exceeds {_MAX_ENV_VALUE_CHARS} characters."
                )
        return value


class ServiceRunResponse(BaseModel):
    """Settled run outcome, returned with HTTP 200 even on nonzero exit or timeout.

    API errors mean the run could not start or be reaped cleanly. Fields mirror
    RunResult plus service_name. The bounded log tail has known sensitive values
    scrubbed and is owner-or-admin gated, like diagnose.last_run.
    """

    service_name: str = Field(description="Stable service name the run executed against")
    run_id: str = Field(description="Identifier of this run (also injected as NERDIT_RUN_ID)")
    exit_code: int | None = Field(
        default=None,
        description=(
            "Container exit code; null only when neither the bounded wait nor "
            "the post-mortem inspect could produce one"
        ),
    )
    timed_out: bool = Field(
        description="Whether the server-side cap fired and the container was killed"
    )
    oom_killed: bool = Field(description="Whether the container was killed by the OOM killer")
    duration_s: float = Field(description="Wall-clock seconds from launch to settle")
    started_at: str = Field(description="ISO timestamp the run started")
    finished_at: str = Field(description="ISO timestamp the run settled")
    log_tail: list[str] = Field(
        default_factory=list,
        description=(
            "Bounded, scrubbed output lines (oldest→newest); combined stdout "
            "and stderr — run output never reaches job_logs (D-P14-6)"
        ),
    )
