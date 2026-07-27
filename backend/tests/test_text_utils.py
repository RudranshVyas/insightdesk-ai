from __future__ import annotations

from backend.app.services import text_utils as T


def test_normalize_for_dedup_strips_ids_and_punctuation() -> None:
    a = T.normalize_for_dedup("Order 88213 failed! Please help.")
    b = T.normalize_for_dedup("order 99999 failed please help")
    assert a == b


def test_normalize_for_lexical_preserves_technical_tokens() -> None:
    out = T.normalize_for_lexical("OTP not received, API returned 500 (ERR-42), v1.2.3")
    assert "otp" in out
    assert "500" in out
    assert "err-42" in out
    assert "v1.2.3" in out


def test_tokenize_lexical_keeps_error_codes_as_single_tokens() -> None:
    toks = T.tokenize_lexical("HTTP 429 rate limit on ERR-77")
    assert "429" in toks
    assert "err-77" in toks


def test_template_groups_collapse_near_duplicates() -> None:
    texts = [
        "Payment failed for order 1001 but the amount was deducted from my card",
        "Payment failed for order 2002 but the amount was deducted from my card",
        "Payment failed for order 3003 but the amount was deducted from my card",
        "The dashboard export returns a 500 server error every single time I try",
    ]
    groups = T.template_groups(texts)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_template_groups_returns_one_id_per_row() -> None:
    texts = ["a" * 40, "b" * 40, ""]
    assert len(T.template_groups(texts)) == 3


def test_short_and_empty_texts_get_their_own_group() -> None:
    groups = T.template_groups(["hi", "hi", ""])
    assert len(set(groups)) == 3


def test_group_size_stats() -> None:
    stats = T.group_size_stats([0, 0, 0, 1, 2])
    assert stats["n_rows"] == 5
    assert stats["n_groups"] == 3
    assert stats["n_multi_member_groups"] == 1
    assert stats["rows_in_multi_member_groups"] == 3
    assert stats["pct_rows_in_multi_member_groups"] == 60.0


def test_unique_ratio_and_labels() -> None:
    assert T.unique_ratio(["a", "a", "b", "c"]) == 0.75
    assert T.repetition_label(0.1) == "high repetition"
    assert T.repetition_label(0.5) == "moderate repetition"
    assert T.repetition_label(0.9) == "low repetition"


def test_unique_ratio_ignores_empty_strings() -> None:
    assert T.unique_ratio(["", "", "a"]) == 1.0
    assert T.unique_ratio([]) == 0.0
