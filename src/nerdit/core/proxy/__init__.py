"""Expose stable HTTPS routes for loopback services through embedded Caddy.

This package re-exports routing, admin, process, TLS, and manager interfaces.
The reconcile loop converges database-backed routes through Caddy's admin API;
inline register/deregister calls are best-effort. Disabled or unavailable Caddy
leaves services on loopback. Settings changes require daemon restart.

Services own default and custom-domain routes. Public ACME issuance requires
both an opted-in domain and enabled ACME settings; all other names use the
internal issuer. Domain ownership and route state come from the database.
"""

from __future__ import annotations

from .admin import CaddyAdmin as CaddyAdmin
from .certs import CertState as CertState
from .certs import CertStatus as CertStatus
from .certs import acme_cert_path as acme_cert_path
from .certs import acme_storage_key as acme_storage_key
from .certs import inspect_cert as inspect_cert
from .domains import DomainInvalid as DomainInvalid
from .domains import ReservedNames as ReservedNames
from .domains import host_aliases as host_aliases
from .domains import normalize_domain as normalize_domain
from .domains import reserved_names as reserved_names
from .domains import validate_domain as validate_domain
from .edgeauth import EdgeAuthInvalid as EdgeAuthInvalid
from .edgeauth import EdgeAuthMaterial as EdgeAuthMaterial
from .edgeauth import EdgeAuthSpec as EdgeAuthSpec
from .edgeauth import auth_fingerprint as auth_fingerprint
from .edgeauth import load_edge_auth as load_edge_auth
from .manager import EdgeAuthUnresolved as EdgeAuthUnresolved
from .manager import ProxyManager as ProxyManager
from .manager import ProxyState as ProxyState
from .routing import _APEX_ID as _APEX_ID
from .routing import _AUTH_UNPARSABLE as _AUTH_UNPARSABLE
from .routing import _ID_PREFIX as _ID_PREFIX
from .routing import LiveRoute as LiveRoute
from .routing import RouteSpec as RouteSpec
from .routing import _domain_route_id as _domain_route_id
from .routing import _extract_auth_fingerprint as _extract_auth_fingerprint
from .routing import _extract_dial as _extract_dial
from .routing import _route_id as _route_id
from .routing import _route_shape as _route_shape
from .routing import _split_route_id as _split_route_id
from .routing import domain_url_for as domain_url_for
from .routing import generate_route as generate_route
from .routing import ordering_violations as ordering_violations
from .routing import public_url_for as public_url_for
from .tls import partition_domains as partition_domains
