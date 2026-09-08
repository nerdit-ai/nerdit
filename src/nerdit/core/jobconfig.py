"""Parse workload config consistently, returning an empty dict for malformed blobs."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def parse_job_config(job: object, *, warn: bool = False) -> dict:
    """Parse raw config JSON or a row's config, returning an empty dict on invalid input.

    Missing/falsy attributes, invalid JSON, and non-object JSON all degrade safely.

    Args:
        warn: Log the job ID on invalid JSON; otherwise remain silent.
    """
    raw = getattr(job, "config", None)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        # ValueError is a strict superset of json.JSONDecodeError, so this one
        # tuple is behaviour-identical to both exception-tuple variants among
        # the predecessors ((TypeError, ValueError) and
        # (TypeError, json.JSONDecodeError)).
        if warn:
            logger.warning("Service %s has invalid config JSON; ignoring", getattr(job, "id", None))
        return {}
    return parsed if isinstance(parsed, dict) else {}
