"""Phase 4 — hybrid retrieval over resolved-ticket PROBLEM descriptions.

Two invariants hold everywhere in this module:

1. The index contains the problem side only. Resolution notes, durations,
   escalation flags, SLA outcomes, and CSAT never enter the dense or lexical
   index — a brand-new ticket has none of them, so indexing them would match on
   information the query cannot possess. They are attached AFTER retrieval.
2. Retrieval strength is computed by the backend from raw dense cosine and rank
   agreement. It is never produced by a language model, and it is never the
   fused RRF score.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.services import text_utils as T

INDEX_VERSION = 1

# Fields attached to a result after retrieval, never indexed.
POST_RETRIEVAL_FIELDS = (
    "resolution_notes",
    "product_area",
    "issue_type",
    "priority",
    "resolution_time_hours",
    "escalated",
    "sla_breached",
    "csat_score",
)

# Fields that must never appear in a retrieval document.
BANNED_FROM_DOCUMENT = (
    "resolution_notes",
    "resolution_time_hours",
    "response_time_hours",
    "escalated",
    "sla_breached",
    "csat_score",
    "status",
    "reopened",
    "resolved_at",
)

METADATA_FIELDS = ("product_area", "issue_type")


# --- retrieval document ------------------------------------------------------


def build_retrieval_document(row: Any) -> str:
    """Problem side only. Metadata lines are omitted when the field is missing."""

    def _get(field_name: str) -> str:
        value = row.get(field_name) if isinstance(row, dict) else getattr(row, field_name, None)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value).strip()

    lines: list[str] = []
    area = _get("product_area")
    if area:
        lines.append(f"Product Area: {area}")
    itype = _get("issue_type")
    if itype:
        lines.append(f"Issue Type: {itype}")

    body = _get("issue_description") or _get("issue_text")
    lines.append(f"Issue: {body}")
    return "\n".join(lines)


def build_query_document(
    issue_description: str, product_area: str | None = None, issue_type: str | None = None
) -> str:
    lines: list[str] = []
    if product_area:
        lines.append(f"Product Area: {product_area}")
    if issue_type:
        lines.append(f"Issue Type: {issue_type}")
    lines.append(f"Issue: {issue_description.strip()}")
    return "\n".join(lines)


# --- FAISS store -------------------------------------------------------------


class FaissTicketStore:
    """FAISS stores vectors and integer positions only. It holds no metadata, so
    the position -> ticket_id list is maintained here and saved beside it."""

    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine, given normalized vectors
        self.pos_to_ticket_id: list[str] = []

    def add(self, ticket_ids: Sequence[str], embeddings: np.ndarray) -> None:
        if len(ticket_ids) != embeddings.shape[0]:
            raise ValueError(
                f"ticket id / embedding count mismatch: "
                f"{len(ticket_ids)} vs {embeddings.shape[0]}"
            )
        self.index.add(np.ascontiguousarray(embeddings.astype("float32")))
        self.pos_to_ticket_id.extend(str(t) for t in ticket_ids)

    def search(self, q: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(top_k, self.index.ntotal)
        scores, idx = self.index.search(
            np.ascontiguousarray(q.reshape(1, -1).astype("float32")), k
        )
        return [
            (self.pos_to_ticket_id[int(i)], float(s))
            for s, i in zip(scores[0], idx[0])
            if i != -1
        ]

    # --- persistence (atomic: index and id map must never disagree) ---------

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        tmp_index = directory / "index.faiss.tmp"
        tmp_ids = directory / "id_map.json.tmp"
        faiss.write_index(self.index, str(tmp_index))
        with open(tmp_ids, "w", encoding="utf-8") as fh:
            json.dump({"dim": self.dim, "pos_to_ticket_id": self.pos_to_ticket_id}, fh)
        os.replace(tmp_index, directory / "index.faiss")
        os.replace(tmp_ids, directory / "id_map.json")

    @classmethod
    def load(cls, directory: Path) -> FaissTicketStore:
        import faiss

        with open(directory / "id_map.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        store = cls.__new__(cls)
        store.dim = int(meta["dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store.pos_to_ticket_id = [str(t) for t in meta["pos_to_ticket_id"]]
        if store.index.ntotal != len(store.pos_to_ticket_id):
            raise ValueError(
                f"index/id-map mismatch: {store.index.ntotal} vectors but "
                f"{len(store.pos_to_ticket_id)} ids. Rebuild the index."
            )
        return store


# --- embeddings --------------------------------------------------------------


def model_slug(name: str, revision: str | None) -> str:
    base = name.replace("/", "__")
    return f"{base}@{revision}" if revision else base


def is_bge(model_name: str) -> bool:
    return "bge" in model_name.lower()


BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Wraps SentenceTransformer with the query/passage asymmetry handled once.

    BGE models expect an instruction prefix on QUERIES ONLY. Applying it to
    corpus documents (or forgetting it on queries) silently destroys recall, so
    the decision lives here rather than at each call site.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.settings = settings or get_settings()
        kwargs: dict[str, Any] = {}
        if self.settings.embedding_model_revision:
            kwargs["revision"] = self.settings.embedding_model_revision
        self.name = self.settings.embedding_model
        self.model = SentenceTransformer(self.name, **kwargs)
        get_dim = getattr(
            self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        self.max_seq_length = int(getattr(self.model, "max_seq_length", 256))

    def encode_documents(self, texts: Sequence[str], show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        payload = (BGE_QUERY_INSTRUCTION + text) if is_bge(self.name) else text
        return self.model.encode(
            [payload], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")[0]

    def truncation_report(self, texts: Sequence[str], sample: int = 5000) -> dict[str, Any]:
        tok = self.model.tokenizer
        sub = list(texts[:sample])
        lengths = [len(tok.encode(t, add_special_tokens=True)) for t in sub]
        over = sum(1 for n in lengths if n > self.max_seq_length)
        return {
            "max_seq_length": self.max_seq_length,
            "sampled": len(sub),
            "pct_truncated": round(100 * over / len(sub), 2) if sub else 0.0,
            "token_p50": int(np.percentile(lengths, 50)) if lengths else 0,
            "token_p95": int(np.percentile(lengths, 95)) if lengths else 0,
            "token_max": int(max(lengths)) if lengths else 0,
        }


def embedding_cache_path(settings: Settings, data_hash: str) -> Path:
    slug = model_slug(settings.embedding_model, settings.embedding_model_revision)
    safe = "".join(c if c.isalnum() or c in "-_.@" else "_" for c in slug)
    return settings.embeddings_dir / f"{safe}__{data_hash[:16]}.npz"


def load_or_compute_embeddings(
    texts: Sequence[str],
    ticket_ids: Sequence[str],
    data_hash: str,
    embedder: Embedder,
    settings: Settings,
    force: bool = False,
) -> tuple[np.ndarray, bool]:
    """Cache keyed by (model, revision, data hash). Never recomputed on restart."""
    path = embedding_cache_path(settings, data_hash)
    if path.exists() and not force:
        blob = np.load(path, allow_pickle=False)
        cached_ids = [str(t) for t in blob["ticket_ids"].tolist()]
        if cached_ids == list(ticket_ids):
            return blob["embeddings"], True
    emb = embedder.encode_documents(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    # np.savez appends ".npz" to a path argument, so write through a file object.
    # Ticket ids are stored as a unicode array, not object dtype, so the cache
    # can be read back with allow_pickle=False.
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, embeddings=emb, ticket_ids=np.array([str(t) for t in ticket_ids]))
    os.replace(tmp, path)
    return emb, False


# --- lexical -----------------------------------------------------------------


class BM25Index:
    def __init__(self, ticket_ids: Sequence[str], tokenized: Sequence[list[str]]) -> None:
        from rank_bm25 import BM25Okapi

        self.ticket_ids = [str(t) for t in ticket_ids]
        self.tokenized = [list(t) for t in tokenized]
        self.bm25 = BM25Okapi(self.tokenized)

    @classmethod
    def build(cls, ticket_ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        return cls(ticket_ids, [T.tokenize_lexical(t) for t in texts])

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = T.tokenize_lexical(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(self.ticket_ids[int(i)], float(scores[int(i)])) for i in top if scores[int(i)] > 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / "bm25.pkl.tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"ticket_ids": self.ticket_ids, "tokenized": self.tokenized}, fh, protocol=4)
        os.replace(tmp, directory / "bm25.pkl")

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        with open(directory / "bm25.pkl", "rb") as fh:
            blob = pickle.load(fh)
        return cls(blob["ticket_ids"], blob["tokenized"])


# --- fusion ------------------------------------------------------------------


def rrf(ranked_lists: Iterable[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion. Fuses by RANK because BM25 scores are unbounded
    while cosine is bounded — combining the raw scores is meaningless without
    calibration nobody has done."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def metadata_boost_unit(k: int, fraction: float) -> float:
    """One RRF unit is 1/(k+1) — with k=60 that is ~0.0164, and two lists cap out
    near 0.033. A naive +0.05 metadata bonus would exceed the entire score range
    and let a product_area match outrank genuine relevance. The boost is a
    deliberate fraction of a single unit."""
    return fraction * (1.0 / (k + 1))


# --- strength gate -----------------------------------------------------------


@dataclass
class StrengthAssessment:
    strength: str  # "strong" | "mixed" | "weak"
    top_cosine: float | None
    margin: float | None
    candidates_above_floor: int
    consensus_above_strong: int
    rank_agreement: int
    metadata_agreement: int
    calibrated: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["note"] = (
            "Retrieval strength is a backend-computed gate over dense cosine and "
            "rank agreement. It is not a probability, not a similarity percentage, "
            "and not produced by a language model."
        )
        if not self.calibrated:
            d["calibration"] = "uncalibrated — conservative defaults in use"
        return d


def assess_strength(
    dense_hits: Sequence[tuple[str, float]],
    lexical_hits: Sequence[tuple[str, float]],
    matched_metadata_counts: dict[str, int],
    settings: Settings | None = None,
) -> StrengthAssessment:
    s = settings or get_settings()
    reasons: list[str] = []

    cosines = [score for _, score in dense_hits]
    top = cosines[0] if cosines else None
    third = cosines[2] if len(cosines) >= 3 else (cosines[-1] if cosines else None)
    margin = (top - third) if (top is not None and third is not None) else None
    above_floor = sum(1 for c in cosines if c >= s.candidate_floor_cosine)
    consensus = sum(1 for c in cosines if c >= s.strong_min_cosine)

    dense_top = {tid for tid, _ in dense_hits[: s.agreement_top_k]}
    lex_top = {tid for tid, _ in lexical_hits[: s.agreement_top_k]}
    agreement = len(dense_top & lex_top)
    meta_agree = sum(1 for v in matched_metadata_counts.values() if v)

    if top is None:
        return StrengthAssessment(
            "weak", None, None, 0, 0, 0, meta_agree, s.retrieval_gate_calibrated,
            ["no dense candidates were returned"],
        )

    strong_cos = top >= s.strong_min_cosine
    strong_margin = margin is not None and margin >= s.strong_min_margin
    enough = above_floor >= s.strong_min_candidates
    strong_consensus = consensus >= s.strong_min_consensus

    if strong_cos and enough and (strong_margin or agreement >= 1 or strong_consensus):
        strength = "strong"
        reasons.append(f"top cosine {top:.3f} >= {s.strong_min_cosine}")
        reasons.append(f"{above_floor} candidates above the {s.candidate_floor_cosine} floor")
        if strong_margin:
            reasons.append(f"top-1 to top-3 margin {margin:.3f} >= {s.strong_min_margin}")
        if agreement:
            reasons.append(f"dense and BM25 agree on {agreement} of the top {s.agreement_top_k}")
        if strong_consensus:
            reasons.append(
                f"{consensus} independent cases clear the {s.strong_min_cosine} cosine bar"
            )
    elif top >= s.mixed_min_cosine and above_floor >= 1:
        strength = "mixed"
        reasons.append(f"top cosine {top:.3f} is between {s.mixed_min_cosine} and {s.strong_min_cosine}")
        if not strong_margin:
            reasons.append("the top result is not clearly separated from the rest")
        if not agreement:
            reasons.append("dense and lexical retrieval do not agree on any candidate")
    else:
        strength = "weak"
        reasons.append(f"top cosine {top:.3f} is below {s.mixed_min_cosine}")
        reasons.append("no evidence is strong enough to justify generating steps")

    return StrengthAssessment(
        strength, round(top, 4), round(margin, 4) if margin is not None else None,
        above_floor, consensus, agreement, meta_agree, s.retrieval_gate_calibrated, reasons,
    )


# --- retriever ---------------------------------------------------------------


@dataclass
class RetrievalResult:
    ticket_id: str
    fusion_rank: int
    fusion_score: float
    dense_rank: int | None
    lexical_rank: int | None
    dense_cosine: float | None
    lexical_score: float | None
    matched_metadata: dict[str, bool]
    attached: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fusion_score_note"] = (
            "Rank-fusion score. Not a similarity and not a percentage."
        )
        return d


class HybridRetriever:
    def __init__(
        self,
        store: FaissTicketStore,
        bm25: BM25Index,
        corpus: pd.DataFrame,
        manifest: dict[str, Any],
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.bm25 = bm25
        self.corpus = corpus.set_index("ticket_id", drop=False)
        self.manifest = manifest
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._embedder

    @classmethod
    def load(cls, settings: Settings | None = None) -> HybridRetriever:
        s = settings or get_settings()
        d = s.retrieval_dir
        with open(d / "manifest.json", encoding="utf-8") as fh:
            manifest = json.load(fh)
        corpus = pd.read_parquet(d / "corpus.parquet")
        return cls(FaissTicketStore.load(d), BM25Index.load(d), corpus, manifest, settings=s)

    # -- search ---------------------------------------------------------------

    def search(
        self,
        issue_description: str,
        product_area: str | None = None,
        issue_type: str | None = None,
        top_k: int = 5,
        mode: str = "hybrid",
        metadata_filter: bool = False,
        metadata_boost: bool | None = None,
        exclude_ticket_ids: set[str] | None = None,
        exclude_template_groups: set[int] | None = None,
    ) -> dict[str, Any]:
        s = self.settings
        query_doc = build_query_document(issue_description, product_area, issue_type)

        dense_hits: list[tuple[str, float]] = []
        lexical_hits: list[tuple[str, float]] = []

        # Over-fetch when excluding, so filtering does not starve the candidate
        # pool and silently shrink the fused list.
        widen = 2 if (exclude_ticket_ids or exclude_template_groups) else 1
        if mode in ("hybrid", "dense"):
            qv = self.embedder.encode_query(query_doc)
            dense_hits = self.store.search(qv, s.dense_candidates * widen)
        if mode in ("hybrid", "lexical"):
            lexical_hits = self.bm25.search(query_doc, s.lexical_candidates * widen)

        # --- evaluation-time exclusions -------------------------------------
        # Phase 5 leave-one-out: a ticket retrieving itself, or one of its own
        # MinHash template siblings, is not evidence of retrieval quality. On
        # templated data it inflates Hit@K into a meaningless number.
        if exclude_ticket_ids or exclude_template_groups:
            def _excluded(tid: str) -> bool:
                if exclude_ticket_ids and tid in exclude_ticket_ids:
                    return True
                if exclude_template_groups and tid in self.corpus.index:
                    group = self.corpus.loc[tid].get("template_group_id")
                    if group is not None and int(group) in exclude_template_groups:
                        return True
                return False

            dense_hits = [h for h in dense_hits if not _excluded(h[0])][: s.dense_candidates]
            lexical_hits = [h for h in lexical_hits if not _excluded(h[0])][
                : s.lexical_candidates
            ]

        if metadata_filter and (product_area or issue_type):
            def _keep(tid: str) -> bool:
                if tid not in self.corpus.index:
                    return False
                row = self.corpus.loc[tid]
                if product_area and str(row.get("product_area")) != product_area:
                    return False
                return not (issue_type and str(row.get("issue_type")) != issue_type)

            dense_hits = [h for h in dense_hits if _keep(h[0])]
            lexical_hits = [h for h in lexical_hits if _keep(h[0])]

        dense_rank = {tid: i for i, (tid, _) in enumerate(dense_hits)}
        lex_rank = {tid: i for i, (tid, _) in enumerate(lexical_hits)}
        dense_score = dict(dense_hits)
        lex_score = dict(lexical_hits)

        lists = []
        if dense_hits:
            lists.append([tid for tid, _ in dense_hits])
        if lexical_hits:
            lists.append([tid for tid, _ in lexical_hits])
        fused = rrf(lists, k=s.rrf_k)

        use_boost = s.metadata_boost_enabled if metadata_boost is None else metadata_boost
        unit = metadata_boost_unit(s.rrf_k, s.metadata_boost_fraction)
        matched: dict[str, dict[str, bool]] = {}
        for tid in list(fused):
            row = self.corpus.loc[tid] if tid in self.corpus.index else None
            m = {
                "product_area": bool(
                    product_area and row is not None and str(row.get("product_area")) == product_area
                ),
                "issue_type": bool(
                    issue_type and row is not None and str(row.get("issue_type")) == issue_type
                ),
            }
            matched[tid] = m
            if use_boost:
                fused[tid] += unit * sum(1 for v in m.values() if v)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        results = [
            RetrievalResult(
                ticket_id=tid,
                fusion_rank=i + 1,
                fusion_score=round(score, 6),
                dense_rank=dense_rank.get(tid) if tid in dense_rank else None,
                lexical_rank=lex_rank.get(tid) if tid in lex_rank else None,
                dense_cosine=round(dense_score[tid], 4) if tid in dense_score else None,
                lexical_score=round(lex_score[tid], 4) if tid in lex_score else None,
                matched_metadata=matched.get(tid, {}),
                attached=self.attach(tid),
            )
            for i, (tid, score) in enumerate(ordered)
        ]
        for r in results:
            if r.dense_rank is not None:
                r.dense_rank += 1
            if r.lexical_rank is not None:
                r.lexical_rank += 1

        meta_counts = {
            f: sum(1 for m in matched.values() if m.get(f)) for f in METADATA_FIELDS
        }
        strength = assess_strength(dense_hits, lexical_hits, meta_counts, s)

        return {
            "results": [r.to_dict() for r in results],
            "strength": strength.to_dict(),
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "k": s.rrf_k,
                "metadata_boost_applied": bool(use_boost),
                "metadata_boost_per_field": round(unit, 6) if use_boost else 0.0,
                "metadata_filter_applied": bool(metadata_filter and (product_area or issue_type)),
                "dense_candidates": len(dense_hits),
                "lexical_candidates": len(lexical_hits),
                "mode": mode,
            },
            "index": {
                "version": self.manifest.get("index_version"),
                "embedding_model": self.manifest.get("embedding_model"),
                "data_hash": self.manifest.get("data_hash"),
                "corpus_size": self.manifest.get("corpus_size"),
            },
        }

    def attach(self, ticket_id: str) -> dict[str, Any]:
        """Attach outcome and resolution fields AFTER retrieval. These were never
        part of the similarity computation."""
        if ticket_id not in self.corpus.index:
            return {}
        row = self.corpus.loc[ticket_id]
        out: dict[str, Any] = {"issue_subject": row.get("issue_subject")}
        # Problem-side fields the brief pipeline needs for display and for
        # template-sibling de-duplication. They were part of the indexed
        # document already, so surfacing them here leaks nothing new.
        for f in ("issue_text", "template_group_id"):
            if f in self.corpus.columns:
                v = row.get(f)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    out[f] = v
        for f in POST_RETRIEVAL_FIELDS:
            if f in self.corpus.columns:
                v = row.get(f)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                out[f] = v.isoformat() if isinstance(v, pd.Timestamp) else v
        return out


# --- corpus construction -----------------------------------------------------


def build_corpus(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from backend.app.services.capabilities import source_case_mask

    mask, info = source_case_mask(df)
    corpus = df[mask].copy()
    corpus["retrieval_document"] = [
        build_retrieval_document(row) for row in corpus.to_dict("records")
    ]
    return corpus, info


def assert_document_is_problem_side_only(documents: Sequence[str], corpus: pd.DataFrame) -> None:
    """Guard rail with teeth: a resolution note leaking into a document would
    make retrieval match on text the query cannot contain."""
    notes = [
        str(n)
        for n in corpus.get("resolution_notes", pd.Series(dtype=str)).fillna("").tolist()
        if len(str(n)) > 40
    ]
    sample_notes = notes[:200]
    joined = "\n".join(documents[:2000])
    for note in sample_notes:
        if note[:40] in joined:
            raise ValueError(
                "a resolution note appears inside a retrieval document; the index "
                "must contain the problem side only"
            )


def data_hash_for(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(str(len(df)).encode())
    for tid in df["ticket_id"].astype(str).tolist():
        h.update(tid.encode())
    for doc in df.get("retrieval_document", pd.Series(dtype=str)).astype(str).tolist():
        h.update(doc.encode("utf-8", "ignore"))
    return h.hexdigest()
