from __future__ import annotations

import pytest

from backend.app.core.redaction import (
    find_placeholders,
    hash_identifier,
    looks_personal,
    redact_text,
    scan_pii,
    strip_placeholders,
)


@pytest.mark.parametrize(
    "raw,token",
    [
        ("write to alice.moreno@example.com now", "[EMAIL]"),
        ("call +1 (415) 555-0132 today", "[PHONE]"),
        ("server at 10.44.12.9 is down", "[IP]"),
        ("key sk-live-77ab99cc11dd22ee failed", "[SECRET]"),
        ("see https://app.example.com/export?report=monthly", "[URL]"),
        ("card 4111 1111 1111 1111 declined", "[CARD]"),
        ("account number ACC-3391045 is locked", "[ACCOUNT]"),
    ],
)
def test_redaction_replaces_pii(raw: str, token: str) -> None:
    out = redact_text(raw)
    assert token in out


def test_redaction_removes_the_original_value() -> None:
    out = redact_text("contact alice.moreno@example.com or 415-555-0177")
    assert "alice.moreno@example.com" not in out
    assert "555-0177" not in out


def test_redaction_is_idempotent() -> None:
    once = redact_text("mail bob@example.com from 10.0.0.1")
    assert redact_text(once) == once


def test_redaction_handles_none_and_non_string() -> None:
    assert redact_text(None) == ""
    assert redact_text(12345) != ""


def test_scan_pii_finds_what_redaction_removes() -> None:
    raw = "email a@b.com and ip 1.2.3.4"
    assert scan_pii(raw)
    assert not scan_pii(redact_text(raw))


def test_placeholder_detection() -> None:
    assert find_placeholders("invoice for {product_purchased} is wrong")
    assert find_placeholders("hello <customer name>")
    assert find_placeholders("account XXXXXX suspended")
    assert not find_placeholders("payment failed for invoice 88213")


def test_placeholder_detection_ignores_redaction_tokens() -> None:
    """Redaction output looks like a template token. It must not be counted."""
    redacted = redact_text("mail alice@example.com about https://x.io/a?b=1")
    assert "[EMAIL]" in redacted
    assert find_placeholders(redacted) == []
    assert "[EMAIL]" in strip_placeholders(redacted)


def test_strip_placeholders_removes_template_tokens_only() -> None:
    out = strip_placeholders("invoice for {product_purchased} missing lines")
    assert "{product_purchased}" not in out
    assert "invoice for" in out


def test_hash_identifier_is_stable_and_non_reversible() -> None:
    a = hash_identifier("alice@example.com")
    b = hash_identifier("alice@example.com")
    c = hash_identifier("bob@example.com")
    assert a == b != c
    assert a is not None and "alice" not in a
    assert hash_identifier(None) is None
    assert hash_identifier("   ") is None


@pytest.mark.parametrize(
    "col", ["Customer Name", "customer_email", "Phone", "Customer Age", "Gender"]
)
def test_looks_personal_flags_protected_columns(col: str) -> None:
    assert looks_personal(col)


@pytest.mark.parametrize("col", ["Ticket Subject", "Product Purchased", "Priority"])
def test_looks_personal_ignores_ordinary_columns(col: str) -> None:
    assert not looks_personal(col)
