"""Projection/resolution seams shared across the `/services` split.

Kept as a leaf package: modules here must never import from
`nerdit.daemon.routes` (the import-cycle scanner counts deferred imports
too), so every `routes/services.py` sub-split imports its shared seams from
here instead of from each other.
"""

from __future__ import annotations
