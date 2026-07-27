"""PII detection and redaction.

Applied before storage, embedding, logging, UI, and any LLM prompt. Order
matters: longer/greedier patterns run first so a URL is not shredded into an
email plus junk.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# --- patterns ---------------------------------------------------------------

_URL_WITH_QUERY = re.compile(r"\bhttps?://[^\s<>\"]+\?[^\s<>\"]*", re.IGNORECASE)
_URL = re.compile(r"\bhttps?://[^\s<>\"]+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Known key shapes first, then a generic long-token fallback.
_API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{16,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})\b"
)
_KEYED_SECRET = re.compile(
    r"\b(api[_\- ]?key|token|password|passwd|secret|bearer)\b\s*[:=]?\s*"
    r"[\"']?([A-Za-z0-9._\-]{8,})[\"']?",
    re.IGNORECASE,
)
# Phone: 10-15 digits with common separators, optional country code.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?"
    r"\d{3,4}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?![\w.])"
)
_CARD = re.compile(r"\b(?:\d[ \-]?){13,19}\b")
_ACCOUNT = re.compile(
    r"\b(?:account|acct|acc|order|invoice|customer)\s*(?:no\.?|number|id|#)?\s*[:#]?\s*"
    r"([A-Za-z]{0,4}[\-]?\d{5,})\b",
    re.IGNORECASE,
)

PLACEHOLDER_TOKENS = re.compile(
    r"(\{\{?[a-z_ ]{2,40}\}?\}"  # {product_purchased} / {{name}}
    r"|<[a-z_ ]{2,40}>"  # <name>
    r"|\[[a-z_ ]{2,40}\]"  # [customer name]
    r"|\bX{4,}\b"  # XXXX
    r"|\blorem ipsum\b)",
    re.IGNORECASE,
)

# Tokens this module itself writes. They look exactly like dataset placeholders,
# so placeholder detection must not count its own output.
REDACTION_TOKENS = frozenset(
    {"[url]", "[email]", "[secret]", "[ip]", "[card]", "[account]", "[phone]"}
)

# Columns that must never survive into stored/canonical data as raw values.
PERSONAL_NAME_HINTS = (
    "name",
    "email",
    "phone",
    "gender",
    "age",
    "address",
    "customer_id",
)


@dataclass
class RedactionReport:
    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, kind: str, n: int = 1) -> None:
        if n:
            self.counts[kind] = self.counts.get(kind, 0) + n

    def total(self) -> int:
        return sum(self.counts.values())


def _sub_count(pattern: re.Pattern[str], repl: str, text: str) -> tuple[str, int]:
    new, n = pattern.subn(repl, text)
    return new, n


def redact_text(text: str | None, report: RedactionReport | None = None) -> str:
    """Return a redacted copy of ``text``. Idempotent on already-redacted text."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    rep = report if report is not None else RedactionReport()

    for kind, pattern, repl in (
        ("url_query", _URL_WITH_QUERY, "[URL]"),
        ("url", _URL, "[URL]"),
        ("email", _EMAIL, "[EMAIL]"),
        ("api_key", _API_KEY, "[SECRET]"),
        ("ip", _IPV4, "[IP]"),
        ("card", _CARD, "[CARD]"),
    ):
        text, n = _sub_count(pattern, repl, text)
        rep.bump(kind, n)

    text, n = _KEYED_SECRET.subn(lambda m: f"{m.group(1)}=[SECRET]", text)
    rep.bump("keyed_secret", n)

    text, n = _ACCOUNT.subn(lambda m: m.group(0).replace(m.group(1), "[ACCOUNT]"), text)
    rep.bump("account", n)

    text, n = _sub_count(_PHONE, "[PHONE]", text)
    rep.bump("phone", n)

    return text


def scan_pii(text: str | None) -> dict[str, int]:
    """Count PII hits without modifying the text. Used by the audit."""
    if not text:
        return {}
    hits = {
        "email": len(_EMAIL.findall(text)),
        "phone": len(_PHONE.findall(text)),
        "ip": len(_IPV4.findall(text)),
        "api_key": len(_API_KEY.findall(text)),
        "url": len(_URL.findall(text)),
        "url_with_query": len(_URL_WITH_QUERY.findall(text)),
        "card": len(_CARD.findall(text)),
        "account": len(_ACCOUNT.findall(text)),
    }
    return {k: v for k, v in hits.items() if v}


def find_placeholders(text: str | None) -> list[str]:
    """Literal template tokens left unfilled by the dataset generator."""
    if not text:
        return []
    return [
        m.group(0)
        for m in PLACEHOLDER_TOKENS.finditer(text)
        if m.group(0).strip().lower() not in REDACTION_TOKENS
    ]


def strip_placeholders(text: str | None) -> str:
    if not text:
        return ""

    def _sub(m: re.Match[str]) -> str:
        return m.group(0) if m.group(0).strip().lower() in REDACTION_TOKENS else " "

    return re.sub(r"\s{2,}", " ", PLACEHOLDER_TOKENS.sub(_sub, text)).strip()


def hash_identifier(value: object, salt: str = "insightdesk") -> str | None:
    """Stable, non-reversible customer id. Never store the raw id."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()[:16]


def looks_personal(column_name: str) -> bool:
    low = column_name.strip().lower().replace(" ", "_")
    return any(h in low for h in PERSONAL_NAME_HINTS)
