"""Does a "resolution note" actually record a resolution?

The measurement that disabled resolution_generation on the shipped corpus, so it
needs its own tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.services import resolution_quality as RQ


@pytest.mark.parametrize(
    "text",
    [
        "We refunded the duplicate charge and enabled idempotency keys.",
        "The issue was resolved after clearing the stale session entry.",
        "Carrier filtering was the root cause was identified and we allow-listed the sender.",
        "I have reset the password and confirmed the customer can sign in.",
    ],
)
def test_completed_actions_are_recognised(text: str) -> None:
    assert RQ.classify(text) == "action"


@pytest.mark.parametrize(
    "text",
    [
        "Could you please provide the exact error message you are seeing?",
        "Thank you for reaching out. Our technical team is actively working on it.",
        "To better assist you, please share more information about your setup.",
        "We apologise for the inconvenience and will investigate further.",
    ],
)
def test_information_requests_are_recognised(text: str) -> None:
    assert RQ.classify(text) == "info_request"


def test_a_reply_that_reports_a_fix_and_asks_a_question_counts_as_an_action() -> None:
    text = "We have refunded the charge. Could you please confirm it arrived?"
    assert RQ.classify(text) == "action"


@pytest.mark.parametrize("text", ["", None, "   "])
def test_empty_is_unclear_not_action(text) -> None:
    assert RQ.classify(text) == "unclear"


def test_a_correspondence_corpus_is_called_correspondence() -> None:
    """The shipped corpus scored 0.4%. The verdict must name it plainly."""
    notes = pd.Series(["Could you please provide more details?"] * 99 + ["We refunded it."])
    out = RQ.assess(notes)
    assert out["verdict"] == "correspondence"
    assert out["action_rate"] == 0.01
    assert "not resolutions" in out["note"]
    assert "would be a claim the data does not support" in out["note"]


def test_a_real_resolution_corpus_passes() -> None:
    notes = pd.Series(["We refunded the duplicate charge."] * 60 + ["Please confirm."] * 40)
    out = RQ.assess(notes)
    assert out["verdict"] == "resolutions"
    assert out["action_rate"] == 0.6


def test_a_mixed_corpus_is_flagged_as_mixed() -> None:
    notes = pd.Series(["We reset the account."] * 10 + ["Could you please confirm?"] * 90)
    out = RQ.assess(notes)
    assert out["verdict"] == "mixed"
    assert "must not be labelled as resolutions without qualification" in out["note"]


def test_empty_corpus_reports_none_not_zero() -> None:
    out = RQ.assess(pd.Series([], dtype=str))
    assert out["action_rate"] is None
    assert out["verdict"] == "no_notes"


def test_blank_notes_are_excluded_from_the_denominator() -> None:
    notes = pd.Series(["We refunded it.", "", "   ", None])
    out = RQ.assess(notes)
    assert out["assessed"] == 1
    assert out["action_rate"] == 1.0
