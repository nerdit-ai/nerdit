"""Resolve declarative Node requirements against Nerdit's pinned runtime images."""

import re
from pathlib import Path

SUPPORTED_NODE_VERSIONS = ("22.23.2", "24.20.0")
DEFAULT_NODE_VERSION = SUPPORTED_NODE_VERSIONS[-1]
_MAX_REQUIREMENT_LENGTH = 512
_VERSION = r"v?(?:0|[1-9][0-9]{0,5}|[xX*])(?:\.(?:0|[1-9][0-9]{0,5}|[xX*])){0,2}"
_TOKEN = re.compile(rf"(>=|<=|>|<|=|\^|~)?\s*({_VERSION})(?=\s|$)")
_ERROR = (
    "Node version requirements are invalid, conflicting, or unsupported. "
    "Use compatible Node 22 or 24 requirements in engines.node, .nvmrc and "
    ".node-version, or provide a Dockerfile for a custom runtime."
)


def _partial(raw: str) -> tuple[tuple[int, int, int], int]:
    pieces = raw.removeprefix("v").split(".")
    numbers: list[int] = []
    wildcard = False
    for piece in pieces:
        if piece in ("x", "X", "*"):
            wildcard = True
        elif wildcard:
            raise ValueError(_ERROR)
        else:
            numbers.append(int(piece))
    padded = numbers + [0, 0, 0]
    return (padded[0], padded[1], padded[2]), len(numbers)


def _upper(version: tuple[int, int, int], precision: int) -> tuple[int, int, int]:
    result = list(version)
    result[precision - 1] += 1
    result[precision:] = [0] * (3 - precision)
    return result[0], result[1], result[2]


def _comparators(operator: str, raw: str) -> list[tuple[str, tuple[int, int, int]]]:
    version, precision = _partial(raw)
    if not precision:
        if operator in ("", "=", "~", "^"):
            return []
        raise ValueError(_ERROR)
    if operator in ("", "="):
        if precision == 3:
            return [("=", version)]
        return [(">=", version), ("<", _upper(version, precision))]
    if operator == "~":
        return [(">=", version), ("<", _upper(version, min(precision, 2)))]
    if operator == "^":
        # A partial zero version permits the remaining unspecified components.
        first_nonzero = next((i + 1 for i, part in enumerate(version) if part), precision)
        return [(">=", version), ("<", _upper(version, first_nonzero))]
    if precision < 3 and operator in (">", "<="):
        return [(">=" if operator == ">" else "<", _upper(version, precision))]
    return [(operator, version)]


def _parse(requirement: str) -> list[list[tuple[str, tuple[int, int, int]]]]:
    if not isinstance(requirement, str) or not 0 < len(requirement) <= _MAX_REQUIREMENT_LENGTH:
        raise ValueError(_ERROR)
    alternatives = []
    for branch in requirement.strip().split("||"):
        branch = branch.strip()
        if not branch:
            raise ValueError(_ERROR)
        hyphen = re.fullmatch(rf"({_VERSION})\s+-\s+({_VERSION})", branch)
        if hyphen:
            low, high = hyphen.groups()
            alternatives.append(_comparators(">=", low) + _comparators("<=", high))
            continue
        comparators = []
        offset = 0
        while offset < len(branch):
            match = _TOKEN.match(branch, offset)
            if match is None:
                raise ValueError(_ERROR)
            operator, raw = match.groups()
            comparators.extend(_comparators(operator or "", raw))
            offset = match.end()
            while offset < len(branch) and branch[offset].isspace():
                offset += 1
        alternatives.append(comparators)
    return alternatives


def _matches(version: tuple[int, int, int], operator: str, required: tuple[int, int, int]) -> bool:
    if operator == "=":
        return version == required
    if operator == ">=":
        return version >= required
    if operator == "<=":
        return version <= required
    if operator == ">":
        return version > required
    return version < required


def resolve_node_version(context: Path, pkg: dict) -> str:
    """Select the newest pinned Node image satisfying every repository declaration.

    Only stable numeric npm ranges are supported; custom aliases or runtimes
    require a Dockerfile. Repository values and filesystem paths stay out of errors.
    """
    requirements = []
    if "engines" in pkg:
        engines = pkg["engines"]
        if not isinstance(engines, dict):
            raise ValueError(_ERROR)
        if "node" in engines:
            requirements.append(_parse(engines["node"]))
    for filename in (".nvmrc", ".node-version"):
        path = context / filename
        try:
            if path.is_symlink():
                raise ValueError(_ERROR)
            if not path.exists():
                continue
            if not path.is_file():
                raise ValueError(_ERROR)
            with path.open("rb") as source:
                raw = source.read(_MAX_REQUIREMENT_LENGTH + 1).decode("utf-8")
        except (OSError, UnicodeError):
            raise ValueError(_ERROR) from None
        if len(raw) > _MAX_REQUIREMENT_LENGTH or not re.fullmatch(_VERSION, raw.strip()):
            raise ValueError(_ERROR)
        requirements.append(_parse(raw.strip()))
    for candidate in reversed(SUPPORTED_NODE_VERSIONS):
        version, _ = _partial(candidate)
        if all(
            any(
                all(_matches(version, operator, required) for operator, required in branch)
                for branch in alternatives
            )
            for alternatives in requirements
        ):
            return candidate
    raise ValueError(_ERROR)
