"""Single place where every artifact version lives.

Every evaluation record, data card, capability manifest, and Support Brief
stamps these. Without them a metric cannot be reproduced, and an unreproducible
metric is not evidence.

Bump a version whenever the *behaviour* of the component changes in a way that
would alter its output on identical input. Bumping is cheap; a silently changed
pipeline that invalidates a published number is not.
"""

from __future__ import annotations

# Column mapping + canonicalization logic in services/schema_adapter.py.
ADAPTER_VERSION = 1

# PII patterns and replacement tokens in core/redaction.py. A bump invalidates
# every stored redacted text and every PII-leakage metric.
REDACTION_VERSION = 1

# Audit computation in services/audit.py.
AUDIT_VERSION = 1

# Capability manifest schema in services/capabilities.py.
MANIFEST_VERSION = 2

# Retrieval index layout (document construction, FAISS store, BM25 payload).
INDEX_VERSION = 1

# Injection / overclaim pattern sets in core/guardrails.py.
GUARDRAIL_VERSION = 1

# Support Brief orchestrator contract in orchestration/.
PIPELINE_VERSION = 1


def version_stamp() -> dict[str, int]:
    """Flat dict of every artifact version, for embedding in outputs."""
    return {
        "adapter": ADAPTER_VERSION,
        "redaction": REDACTION_VERSION,
        "audit": AUDIT_VERSION,
        "manifest": MANIFEST_VERSION,
        "index": INDEX_VERSION,
        "guardrail": GUARDRAIL_VERSION,
        "pipeline": PIPELINE_VERSION,
    }
