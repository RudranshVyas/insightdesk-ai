"""Text normalization, fingerprinting, and near-duplicate template grouping.

Near-duplicate detection uses MinHash + LSH. All-pairs cosine over 200K rows is
not an option and is never attempted here.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence

from datasketch import MinHash, MinHashLSH

_WS = re.compile(r"\s+")
_NUMERIC_ID = re.compile(r"\b[a-z]*\d[\w\-]*\b")
_PUNCT = re.compile(r"[^\w\s]")

# Tokens that must survive lexical normalization: error codes, protocol nouns,
# product identifiers. Stripping punctuation blindly destroys these.
# Alternation order matters: the compound patterns must be tried before the bare
# acronym pattern, or `ERR-42` is split into `ERR` + `-42`.
_TECHNICAL_TOKEN = re.compile(
    r"\b(?:[A-Za-z]+[_\-]\d+"  # ERR-500, code_42
    r"|v\d+(?:\.\d+)+"  # v1.2.3
    r"|[A-Z]{2,6}\d{0,4}"  # OTP, API, SAML, HTTP2
    r"|\d{3})\b"  # HTTP status codes
)


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", str(text)).strip()


def normalize_for_dedup(text: str | None) -> str:
    """Aggressive normalization used ONLY for duplicate/near-duplicate stats.

    Lowercase, collapse whitespace, strip punctuation and numeric identifiers.
    Never use this for embedding or lexical indexing — it destroys error codes.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = _NUMERIC_ID.sub(" ", t)
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def normalize_for_lexical(text: str | None) -> str:
    """Unicode-normalize and lowercase while preserving technical tokens."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    preserved: list[str] = []

    def _stash(m: re.Match[str]) -> str:
        preserved.append(m.group(0))
        return f" \x00{len(preserved) - 1}\x00 "

    t = _TECHNICAL_TOKEN.sub(_stash, t)
    t = t.lower()
    t = re.sub(r"[^\w\s\x00./#\-]", " ", t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: preserved[int(m.group(1))].lower(), t)
    return _WS.sub(" ", t).strip()


def tokenize_lexical(text: str | None) -> list[str]:
    return [t for t in normalize_for_lexical(text).split(" ") if t]


def char_shingles(text: str, n: int = 5) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _minhash(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in char_shingles(text, 5):
        m.update(sh.encode("utf-8"))
    return m


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def template_groups(
    texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 64,
    min_chars: int = 20,
) -> list[int]:
    """Assign every text a ``template_group_id`` via MinHash-LSH + union-find.

    Texts shorter than ``min_chars`` and empty texts each get their own group —
    they are too short for shingle similarity to mean anything.

    **Exact duplicates are collapsed before MinHash runs, and this is load-bearing
    rather than an optimization.** LSH is near-linear only while buckets stay
    small. On a heavily templated support corpus the opposite is true: a real
    200K-row dataset here carried roughly ten distinct issue descriptions, so
    every row was a near-duplicate of every other, each ``lsh.query`` returned
    tens of thousands of candidates, and the union pass became O(n^2) — billions
    of operations that never complete. Deduplicating first turns that same input
    into ~10 signatures. Grouping identical text together is also exactly what
    the caller means by a template group, so nothing is lost.

    Returns a list of group ids parallel to ``texts``.
    """
    n = len(texts)
    normed = [normalize_for_dedup(t) for t in texts]

    # --- collapse exact normalized duplicates --------------------------------
    representative: dict[str, int] = {}
    members: list[list[int]] = []
    short_or_empty: list[int] = []

    for i, t in enumerate(normed):
        if len(t) < min_chars:
            short_or_empty.append(i)
            continue
        slot = representative.get(t)
        if slot is None:
            slot = len(members)
            representative[t] = slot
            members.append([])
        members[slot].append(i)

    distinct = list(representative.items())  # (normalized_text, slot)

    # --- MinHash only the distinct texts -------------------------------------
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures: dict[int, MinHash] = {}
    for text, slot in distinct:
        m = _minhash(text, num_perm)
        signatures[slot] = m
        lsh.insert(str(slot), m)

    uf = _UnionFind(len(members))
    for slot, m in signatures.items():
        for other in lsh.query(m):
            uf.union(slot, int(other))

    # --- expand back to one group id per input row ---------------------------
    remap: dict[int, int] = {}
    out: list[int] = [0] * n

    for slot in range(len(members)):
        root = uf.find(slot)
        if root not in remap:
            remap[root] = len(remap)

    for slot, rows in enumerate(members):
        gid = remap[uf.find(slot)]
        for row in rows:
            out[row] = gid

    # Texts too short to fingerprint each get their own singleton group, so they
    # are never silently merged with anything.
    next_gid = len(remap)
    for row in short_or_empty:
        out[row] = next_gid
        next_gid += 1

    return out


def group_size_stats(group_ids: Sequence[int]) -> dict[str, object]:
    sizes: dict[int, int] = defaultdict(int)
    for g in group_ids:
        sizes[g] += 1
    counts = sorted(sizes.values(), reverse=True)
    total = len(group_ids)
    multi = [c for c in counts if c > 1]
    top10 = counts[:10]
    return {
        "n_rows": total,
        "n_groups": len(counts),
        "n_multi_member_groups": len(multi),
        "rows_in_multi_member_groups": sum(multi),
        "pct_rows_in_multi_member_groups": round(100 * sum(multi) / total, 2) if total else 0.0,
        "largest_group_sizes": top10,
        "pct_rows_in_top10_groups": round(100 * sum(top10) / total, 2) if total else 0.0,
    }


def unique_ratio(values: Iterable[str]) -> float:
    vals = [v for v in values if v]
    if not vals:
        return 0.0
    return round(len(set(vals)) / len(vals), 4)


def repetition_label(ratio: float) -> str:
    """Diagnostic label only. Not a pass/fail judgement."""
    if ratio < 0.3:
        return "high repetition"
    if ratio < 0.7:
        return "moderate repetition"
    return "low repetition"
