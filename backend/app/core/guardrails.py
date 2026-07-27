"""Guardrail primitives shared by ingestion (Phase 1) and the brief pipeline
(Phase 6).

Two independent concerns live here:

1. **Injection detection.** Ticket text and resolution notes are untrusted data.
   They are never instructions. This module does not "sanitize" them into
   safety — that is not achievable — it *flags* them, and the pipeline wraps
   them in explicit data delimiters so the model is told what they are.
2. **Overclaim detection.** A resolution suggestion built from historical
   evidence may never promise a guaranteed outcome. Generated text is scanned
   for certainty language and the offending phrases are reported.

Both are deterministic regex passes. Neither calls a model, so both keep working
with ``LLM_PROVIDER=none``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- injection ---------------------------------------------------------------

# Each entry is (label, pattern). Labels are stable identifiers used in metrics
# and evaluation records, so renaming one is a breaking change to the reports.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}?"
            r"\b(?:previous|prior|above|earlier|all|any|the)\b[^.\n]{0,20}?"
            r"\b(?:instruction|instructions|prompt|prompts|rule|rules|context|"
            r"direction|directions|guideline|guidelines)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_probe",
        re.compile(
            r"\b(?:system\s+prompt|initial\s+prompt|your\s+instructions|"
            r"reveal\s+your|print\s+your|repeat\s+your|show\s+me\s+your)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you|act\s+as\s+(?:a|an|the)|"
            r"pretend\s+to\s+be|roleplay\s+as|assume\s+the\s+role)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_token_injection",
        # A chat role marker at the start of a line is never legitimate ticket
        # prose; it is an attempt to forge a turn boundary.
        re.compile(
            r"(?:^|\n)\s*(?:###\s*)?(?:system|assistant|user|human)\s*[:>]",
            re.IGNORECASE,
        ),
    ),
    (
        "delimiter_escape",
        re.compile(
            r"(?:</?(?:evidence|instruction|system|prompt|context)\s*>"
            r"|\[/?(?:INST|SYS|EVIDENCE)\]"
            r"|<\|[a-z_]{2,20}\|>)",
            re.IGNORECASE,
        ),
    ),
    (
        "exfiltration_request",
        # Detection runs on REDACTED text, so the destination has usually already
        # become `[URL]` or `[EMAIL]` by the time this pattern sees it. Matching
        # only on a literal `https://` would miss every real attempt.
        # The middle group is what separates an attack from an ordinary
        # "email me at [EMAIL]": the thing being sent must be the model's own
        # context, not the customer's information.
        re.compile(
            r"\b(?:send|post|upload|email|forward|exfiltrate|leak)\b[^.\n]{0,25}?"
            r"\b(?:context|prompt|instructions?|conversation|chat\s+history|"
            r"system\s+message|everything|all\s+(?:the\s+)?(?:data|tickets?|records?)|"
            r"contents?|every\s+ticket)\b[^.\n]{0,40}?"
            r"(?:\[URL\]|\[EMAIL\]|https?://|[\w.\-]+@[\w.\-]+"
            r"|\bwebhook\b|\bendpoint\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_or_capability_coercion",
        re.compile(
            r"\b(?:call\s+the\s+tool|execute|run)\b[^.\n]{0,25}?"
            r"\b(?:command|shell|script|sql|query)\b"
            r"|\bdelete\s+(?:all|every|the)\b[^.\n]{0,20}?\b(?:record|ticket|row|data)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "urgency_authority_claim",
        re.compile(
            r"\b(?:as\s+(?:an?\s+)?(?:admin|administrator|developer|engineer|"
            r"anthropic|openai)\b"
            r"|this\s+is\s+(?:a\s+)?(?:test\s+mode|debug\s+mode|authorized)\b"
            r"|you\s+(?:are|have\s+been)\s+(?:authorized|permitted|allowed)\b)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass
class InjectionScan:
    """Result of scanning one piece of untrusted text."""

    flagged: bool
    labels: list[str] = field(default_factory=list)
    # Matched spans, truncated. Safe to log: they are the attack string, not the
    # customer's private content. Still capped so a log line cannot be flooded.
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"flagged": self.flagged, "labels": self.labels, "samples": self.samples}


def scan_injection(text: str | None, max_samples: int = 5) -> InjectionScan:
    """Flag instruction-like patterns in untrusted text.

    Detection is advisory. Nothing downstream may treat "not flagged" as proof
    the text is benign — the defence is that evidence is *structurally* placed
    in a data section, never an instruction section.
    """
    if not text:
        return InjectionScan(False)

    labels: list[str] = []
    samples: list[str] = []
    for label, pattern in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            labels.append(label)
            if len(samples) < max_samples:
                samples.append(m.group(0).strip()[:120])
    return InjectionScan(bool(labels), labels, samples)


def neutralize_delimiters(text: str | None) -> str:
    """Break literal delimiter tokens so untrusted text cannot close the data
    block it is nested inside.

    This is defence in depth, not a sanitizer. It only defuses the exact tokens
    this codebase uses to fence evidence.
    """
    if not text:
        return ""
    out = re.sub(r"<(/?)(evidence|instruction|system|prompt|context)\s*>", r"<\1 \2>", str(text), flags=re.IGNORECASE)
    out = re.sub(r"\[(/?)(INST|SYS|EVIDENCE)\]", r"[\1 \2]", out, flags=re.IGNORECASE)
    out = re.sub(r"<\|([a-z_]{2,20})\|>", r"< |\1| >", out, flags=re.IGNORECASE)
    return out


# --- overclaiming ------------------------------------------------------------

# A support brief presents historical evidence. Any phrasing that promises an
# outcome misrepresents what the system can know.
_OVERCLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("guarantee", re.compile(r"\b(?:guarantee[sd]?|guaranteed)\b", re.IGNORECASE)),
    # `100\s*%` sits outside the \b-terminated alternation: `%` is not a word
    # character, so a trailing \b would never match after it.
    ("certainty", re.compile(r"100\s*%"
                             r"|\b(?:definitely|certainly|undoubtedly|without\s+(?:a\s+)?doubt"
                             r"|absolutely\s+(?:will|works))\b", re.IGNORECASE)),
    ("always_never", re.compile(r"\b(?:always\s+(?:works|fixes|resolves|solves)|"
                                r"never\s+fails|cannot\s+fail)\b", re.IGNORECASE)),
    ("risk_free", re.compile(r"\b(?:risk[\s\-]?free|no\s+risk|completely\s+safe|"
                             r"fully\s+automatic(?:ally)?\s+resolv)\b", re.IGNORECASE)),
    ("auto_resolution", re.compile(r"\bautomatic(?:ally)?\s+(?:resolve[sd]?|resolution)\b",
                                   re.IGNORECASE)),
    ("will_definitely_fix", re.compile(r"\bthis\s+will\s+(?:fix|resolve|solve)\b", re.IGNORECASE)),
)


@dataclass
class OverclaimScan:
    flagged: bool
    labels: list[str] = field(default_factory=list)
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {"flagged": self.flagged, "labels": self.labels, "samples": self.samples}


def scan_overclaiming(text: str | None, max_samples: int = 5) -> OverclaimScan:
    if not text:
        return OverclaimScan(False)
    labels: list[str] = []
    samples: list[str] = []
    for label, pattern in _OVERCLAIM_PATTERNS:
        m = pattern.search(text)
        if m:
            labels.append(label)
            if len(samples) < max_samples:
                samples.append(m.group(0).strip()[:80])
    return OverclaimScan(bool(labels), labels, samples)
