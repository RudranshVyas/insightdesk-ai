"""Does a "resolution note" actually record a resolution?

A column named `resolution_notes` is a promise the data does not always keep.
Many support exports carry the *first agent reply* under that name, and a first
reply is usually an acknowledgement or a request for more information — not a
record of what fixed the problem.

This matters more than it sounds. If the corpus is mostly information requests,
then a UI header reading "what resolved it" is a fabricated claim, and a
retrieval system built on it returns correspondence rather than solutions. The
measured rate on the shipped Kaggle corpus was **0.1% action-bearing**, which is
why `resolution_generation` disables itself on that data.

The classifier is a deliberately simple lexical heuristic. It is not trying to
be clever; it is trying to produce an auditable number that a human can sanity
check by reading twenty rows. It reports a rate, and the capability layer decides
what that rate is worth.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Phrases that indicate the agent is asking the customer for something, or
# acknowledging receipt, rather than reporting an action they took.
ASKS_FOR_INFO = re.compile(
    r"\b(?:"
    r"could you (?:please )?(?:provide|share|let me know|confirm|send|specify)"
    r"|please (?:provide|share|confirm|send|let us know|specify|advise)"
    r"|can you (?:please )?(?:provide|share|confirm|send)"
    r"|kindly (?:provide|share|confirm)"
    r"|(?:more|additional|further) (?:information|details)"
    r"|to (?:better )?assist you"
    r"|we (?:will|are) (?:currently )?(?:investigat|look|review|work)"
    r"|our (?:technical )?team is (?:actively )?(?:working|investigating)"
    r"|thank you for (?:reaching out|contacting)"
    r"|we (?:apologi[sz]e|regret)"
    r"|what (?:operating system|browser|version|error)"
    r")",
    re.IGNORECASE,
)

# Verbs that name a completed action. Kept separate because agents write notes
# subject-less and past-tense — "Voided the hold.", "Refunded the second charge."
# — so requiring a leading "we" misses the most common phrasing there is.
_ACTION_VERBS = (
    r"refunded|voided|reset|restored|corrected|updated|enabled|disabled|replaced"
    r"|reissued|escalated|applied|cleared|fixed|repaired|migrated|configured"
    r"|allow-?listed|whitelisted|reinstalled|patched|credited|switched|moved"
    r"|increased|decreased|raised|lowered|removed|deleted|added|installed"
    r"|rebooted|restarted|rotated|renewed|cancelled|canceled|processed|deployed"
)

# Phrases that indicate a completed action — something was actually done.
DESCRIBES_ACTION = re.compile(
    r"(?:"
    # Sentence-initial past-tense verb: the house style of a real agent note.
    rf"(?:^|(?<=[.!?]\s)|(?<=[.!?]\s\s))\s*(?:{_ACTION_VERBS})\b"
    # Coordinated second action: "...and refunded the charge".
    rf"|\b(?:and|then|so)\s+(?:{_ACTION_VERBS})\b"
    r"|\b(?:"
    r"(?:we|i) (?:have )?(?:refunded|voided|reset|restored|corrected|updated|enabled"
    r"|disabled|replaced|reissued|escalated|applied|cleared|fixed|repaired|migrated"
    r"|configured|allow-?listed|whitelisted|rolled back|reinstalled|patched|credited)"
    r"|(?:has|have|was|were) been (?:refunded|reset|restored|corrected|resolved|fixed"
    r"|replaced|reissued|credited|applied|updated|patched)"
    r"|(?:issue|problem|error|outage) (?:was|has been|is now) (?:resolved|fixed|corrected)"
    r"|resolved (?:by|after|following)"
    r"|(?:the )?(?:root )?cause was"
    r"|turned out to be"
    r"|fix(?:ed)? (?:was )?(?:deployed|shipped|released)"
    r"))",
    re.IGNORECASE | re.MULTILINE,
)


def classify(text: str | None) -> str:
    """One of: `action`, `info_request`, `unclear`.

    `action` wins ties: a reply that both reports a fix and asks a follow-up
    question still records a resolution.
    """
    if not text or not str(text).strip():
        return "unclear"
    s = str(text)
    if DESCRIBES_ACTION.search(s):
        return "action"
    if ASKS_FOR_INFO.search(s):
        return "info_request"
    return "unclear"


def assess(notes: pd.Series) -> dict[str, Any]:
    """Measure what a corpus of resolution notes actually contains."""
    values = notes.fillna("").astype(str)
    values = values[values.str.strip() != ""]
    n = int(len(values))

    if n == 0:
        return {
            "assessed": 0,
            "action_count": 0,
            "info_request_count": 0,
            "unclear_count": 0,
            "action_rate": None,
            "verdict": "no_notes",
            "note": "No non-empty resolution notes exist to assess.",
        }

    labels = values.map(classify)
    action = int((labels == "action").sum())
    info = int((labels == "info_request").sum())
    unclear = int((labels == "unclear").sum())
    rate = round(action / n, 4)

    if rate >= 0.20:
        verdict = "resolutions"
        note = (
            f"{action} of {n} notes describe an action that was taken. The column "
            f"behaves like a resolution record."
        )
    elif rate >= 0.05:
        verdict = "mixed"
        note = (
            f"Only {action} of {n} notes ({rate:.1%}) describe an action taken; "
            f"{info} ask the customer for information. The column is a mix of "
            f"resolutions and correspondence, and must not be labelled as "
            f"resolutions without qualification."
        )
    else:
        verdict = "correspondence"
        note = (
            f"Only {action} of {n} notes ({rate:.1%}) describe an action taken, "
            f"while {info} ask the customer for information. This column holds "
            f"support CORRESPONDENCE, not resolutions. Presenting it as 'what "
            f"resolved the ticket' would be a claim the data does not support."
        )

    return {
        "assessed": n,
        "action_count": action,
        "info_request_count": info,
        "unclear_count": unclear,
        "action_rate": rate,
        "verdict": verdict,
        "note": note,
        "method": (
            "Lexical heuristic over completed-action phrasing versus "
            "information-request phrasing. Deliberately simple so the number can "
            "be sanity checked by reading a sample."
        ),
    }
