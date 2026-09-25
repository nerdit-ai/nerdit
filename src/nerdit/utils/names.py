"""Shared name grammars: the one DNS-label and volume-name regex."""

from __future__ import annotations

import re

#: A lowercase RFC 1123 label, 1-63 chars: service names, hosted slugs, the
#: on-disk scope names under `<data_dir>`.
DNS_LABEL_PATTERN = r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$"
#: Always use `fullmatch`: with `match`, `$` admits a trailing newline.
DNS_LABEL_RE = re.compile(DNS_LABEL_PATTERN)

#: A `[deploy].volumes` name, 1-32 chars, same alphabet as a DNS label.
VOLUME_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$")
