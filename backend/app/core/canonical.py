"""Canonical ticket schema.

Every dataset-specific column name lives in a mapping YAML. No other module in
this app may reference a source column name.
"""

from __future__ import annotations

# Only these two are required for the app to do anything at all.
REQUIRED_FIELDS: tuple[str, ...] = ("ticket_id", "issue_text")

CANONICAL_FIELDS: tuple[str, ...] = (
    "ticket_id",
    "created_at",
    "resolved_at",
    "customer_id_hash",
    "customer_segment",
    "product_area",
    "issue_type",
    "priority",
    "status",
    "channel",
    "platform",
    "region",
    "sla_plan",
    "sla_deadline",
    "issue_subject",
    "issue_description",
    "issue_text",
    "resolution_notes",
    "first_response_at",
    "response_time_hours",
    "resolution_time_hours",
    "reopened",
    "escalated",
    "sla_breached",
    "csat_score",
)

TIMESTAMP_FIELDS: tuple[str, ...] = (
    "created_at",
    "resolved_at",
    "sla_deadline",
    "first_response_at",
)

BOOLEAN_FIELDS: tuple[str, ...] = ("reopened", "escalated", "sla_breached")

NUMERIC_FIELDS: tuple[str, ...] = (
    "response_time_hours",
    "resolution_time_hours",
    "csat_score",
)

TEXT_FIELDS: tuple[str, ...] = (
    "issue_subject",
    "issue_description",
    "issue_text",
    "resolution_notes",
)

CATEGORICAL_FIELDS: tuple[str, ...] = (
    "customer_segment",
    "product_area",
    "issue_type",
    "priority",
    "status",
    "channel",
    "platform",
    "region",
    "sla_plan",
)

# Candidate outcome targets for the Phase 8 risk model, in ladder order.
OUTCOME_CANDIDATES: tuple[str, ...] = (
    "escalated",
    "sla_breached",
    "resolution_time_hours",
    "csat_score",
)

# Never usable as a model feature: post-outcome, hindsight, or protected.
FORBIDDEN_RISK_FEATURES: frozenset[str] = frozenset(
    {
        "resolution_notes",
        "resolved_at",
        "resolution_time_hours",
        "response_time_hours",
        "first_response_at",
        "first_response_time_hours",
        "csat_score",
        "status",
        "reopened",
        "escalated",
        "sla_breached",
        "customer_id",
        "customer_id_hash",
        "customer_name",
        "customer_email",
        "email",
        "phone",
        "gender",
        "age",
        "customer_gender",
        "customer_age",
        "customer_sentiment",
    }
)

# Normalized status vocabulary. Mapping file decides which raw values land here.
STATUS_VALUES: tuple[str, ...] = ("open", "pending", "resolved", "closed", "cancelled")

# Resolution-note strings that carry no information and must not qualify a
# ticket as a retrieval source case.
BOILERPLATE_RESOLUTIONS: frozenset[str] = frozenset(
    {
        "",
        "n/a",
        "na",
        "nan",
        "none",
        "null",
        "-",
        "--",
        "resolved",
        "closed",
        "done",
        "fixed",
        "ok",
        "no",
        "nil",
        "not applicable",
        "no resolution",
        "no notes",
        "test",
    }
)
