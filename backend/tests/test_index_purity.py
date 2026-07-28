"""The retrieval-index purity guard.

A CI quality gate, so its own failure modes matter. It must fire on real
leakage and stay silent on the vocabulary overlap that every natural-language
support corpus contains.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.services import retrieval as R


def _corpus(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_clean_corpus_passes() -> None:
    corpus = _corpus([
        {"ticket_id": "T1", "resolution_notes": "We voided the duplicate authorization hold and refunded the customer within five business days."},
        {"ticket_id": "T2", "resolution_notes": "Carrier filtering blocked the SMS sender id, so we moved the customer to email OTP delivery."},
    ])
    docs = [
        "Product Area: Payments\nIssue: I was charged twice for one invoice.",
        "Product Area: Auth\nIssue: The one-time passcode never arrives by SMS.",
    ]
    R.assert_document_is_problem_side_only(docs, corpus)


def test_a_ticket_carrying_its_own_resolution_note_is_caught() -> None:
    note = "We voided the duplicate authorization hold and refunded the customer within five business days."
    corpus = _corpus([{"ticket_id": "T1", "resolution_notes": note}])
    docs = [f"Product Area: Payments\nIssue: I was charged twice. {note}"]
    with pytest.raises(ValueError, match="its own resolution note"):
        R.assert_document_is_problem_side_only(docs, corpus)


def test_the_error_names_the_offending_ticket() -> None:
    note = "Carrier filtering blocked the SMS sender id, so we moved the customer to email OTP delivery."
    corpus = _corpus([
        {"ticket_id": "T1", "resolution_notes": "unrelated but long enough to be a distinctive fingerprint here"},
        {"ticket_id": "T2", "resolution_notes": note},
    ])
    docs = ["Issue: clean", f"Issue: leaked {note}"]
    with pytest.raises(ValueError, match="T2"):
        R.assert_document_is_problem_side_only(docs, corpus)


def test_shared_stock_phrases_across_different_tickets_do_not_trip_it() -> None:
    """The regression this guard was rewritten for.

    Agents and customers write the same boilerplate. Ticket A's answer opening
    appearing inside ticket B's question is ordinary English, not leakage — the
    query for B genuinely could contain that phrase.
    """
    stock = "Could you please provide details about your account configuration so we can help"
    corpus = _corpus([
        {"ticket_id": "T1", "resolution_notes": stock + " you further with this issue."},
        {"ticket_id": "T2", "resolution_notes": "A completely different resolution that shares no opening text at all."},
    ])
    docs = [
        "Issue: My account is locked and I cannot sign in to the portal.",
        # T2's *question* contains T1's *answer* opening. Different tickets.
        f"Issue: {stock} me resolve the billing discrepancy on my latest statement.",
    ]
    R.assert_document_is_problem_side_only(docs, corpus)


def test_short_notes_are_skipped_rather_than_matched_by_luck() -> None:
    """A 10-character note would match half the corpus by coincidence."""
    corpus = _corpus([{"ticket_id": "T1", "resolution_notes": "Fixed."}])
    docs = ["Issue: Fixed. The thing was broken and then it was Fixed."]
    R.assert_document_is_problem_side_only(docs, corpus)


def test_missing_resolution_column_is_not_an_error() -> None:
    corpus = _corpus([{"ticket_id": "T1"}])
    R.assert_document_is_problem_side_only(["Issue: anything"], corpus)


def test_banned_outcome_field_in_the_template_is_caught() -> None:
    """Catches the class the per-row probe cannot: a short outcome value added
    to the document template."""
    corpus = _corpus([{"ticket_id": "T1", "resolution_notes": "x" * 200}])
    docs = ["Product Area: Payments\nEscalated: True\nIssue: something"]
    with pytest.raises(ValueError, match="banned outcome fields"):
        R.assert_document_is_problem_side_only(docs, corpus)


def test_the_real_document_builder_produces_a_clean_document() -> None:
    """End to end: what build_retrieval_document actually emits must pass."""
    row = {
        "ticket_id": "T1",
        "product_area": "Payments",
        "issue_type": "Incident",
        "issue_description": "I was charged twice for the same invoice.",
        "resolution_notes": "We voided the duplicate hold and refunded within five business days.",
        "escalated": True,
        "csat_score": 5,
    }
    doc = R.build_retrieval_document(row)
    for banned_value in ("voided the duplicate", "True", "5"):
        assert banned_value not in doc or banned_value == "5" and "5" not in doc
    assert "resolution" not in doc.lower()
    R.assert_document_is_problem_side_only([doc], _corpus([row]))
