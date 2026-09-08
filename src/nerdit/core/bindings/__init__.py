"""Resource-binding metadata for diagnostics and shared secret resolution.

`BINDING_KINDS` describes injected key names and shared-secret fields for each
binding kind. Launch and write paths use kind-specific parsers, resolvers, and
injectors directly. `secretref` provides their common reference resolver.
"""
