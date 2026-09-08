"""Pydantic request and response schemas, grouped by route domain.

Persistence rows and enums live in db.rows and db.enums. Never import the
db.models re-export module here: it imports these schemas and would cycle.
"""
