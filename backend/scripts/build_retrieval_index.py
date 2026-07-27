"""Phase 4 — build the retrieval artifacts.

Explicit offline command. The API loads what this writes and never builds it.

Writes to artifacts/retrieval/:
  index.faiss      dense vectors
  id_map.json      faiss position -> ticket_id (FAISS itself stores no metadata)
  bm25.pkl         tokenized lexical corpus
  corpus.parquet   source cases plus the exact indexed document
  manifest.json    model, data hash, index version, fusion settings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from backend.app.core.capability_loader import load_capabilities
from backend.app.core.config import get_settings
from backend.app.services import retrieval as R


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Build the hybrid retrieval index.")
    p.add_argument("--parquet", type=Path, default=settings.tickets_parquet)
    p.add_argument("--sample", type=int, default=None, help="index only N source cases (dev)")
    p.add_argument("--force-embeddings", action="store_true", help="ignore the embedding cache")
    p.add_argument("--ignore-capabilities", action="store_true")
    args = p.parse_args(argv)

    caps = load_capabilities()
    if not args.ignore_capabilities and not caps.get("retrieval", {}).get("enabled"):
        print(
            "retrieval is disabled by the capability manifest:\n  "
            f"{caps.get('retrieval', {}).get('reason')}",
            file=sys.stderr,
        )
        return 3

    if not args.parquet.exists():
        print(f"missing {args.parquet}; run ingest_tickets first", file=sys.stderr)
        return 2

    settings.ensure_dirs()
    df = pd.read_parquet(args.parquet)
    corpus, info = R.build_corpus(df)
    print(f"Source cases: {info['eligible_source_cases']} of {len(df)} tickets")
    if info.get("relaxation"):
        print(f"  ! {info['relaxation']}")
    if args.sample:
        corpus = corpus.head(args.sample)
        print(f"  --sample: indexing {len(corpus)} cases")

    if corpus.empty:
        print("no eligible source cases; nothing to index", file=sys.stderr)
        return 3

    documents = corpus["retrieval_document"].tolist()
    R.assert_document_is_problem_side_only(documents, corpus)
    ticket_ids = corpus["ticket_id"].astype(str).tolist()
    data_hash = R.data_hash_for(corpus)
    print(f"Data hash: {data_hash[:16]}")

    print(f"Loading embedding model {settings.embedding_model} ...")
    embedder = R.Embedder(settings)
    trunc = embedder.truncation_report(documents)
    print(
        f"  tokens p50/p95/max {trunc['token_p50']}/{trunc['token_p95']}/{trunc['token_max']}, "
        f"{trunc['pct_truncated']}% truncated at {trunc['max_seq_length']}"
    )

    t0 = time.perf_counter()
    embeddings, cached = R.load_or_compute_embeddings(
        documents, ticket_ids, data_hash, embedder, settings, force=args.force_embeddings
    )
    print(
        f"  embeddings {'loaded from cache' if cached else 'computed'} "
        f"({embeddings.shape[0]}x{embeddings.shape[1]}) in {time.perf_counter() - t0:.1f}s"
    )

    store = R.FaissTicketStore(embeddings.shape[1])
    store.add(ticket_ids, embeddings)
    store.save(settings.retrieval_dir)

    print("Building BM25 ...")
    bm25 = R.BM25Index.build(ticket_ids, documents)
    bm25.save(settings.retrieval_dir)

    keep = [
        c
        for c in (
            "ticket_id", "retrieval_document", "issue_subject", "issue_text",
            "template_group_id", *R.POST_RETRIEVAL_FIELDS,
        )
        if c in corpus.columns
    ]
    corpus[keep].to_parquet(settings.retrieval_dir / "corpus.parquet", index=False)

    manifest = {
        "index_version": R.INDEX_VERSION,
        "built_at": datetime.now(UTC).isoformat(),
        "embedding_model": settings.embedding_model,
        "embedding_model_revision": settings.embedding_model_revision,
        "embedding_dim": int(embeddings.shape[1]),
        "data_hash": data_hash,
        "corpus_size": len(corpus),
        "source_case_selection": info,
        "truncation": trunc,
        "document_template": "Product Area / Issue Type / Issue — problem side only",
        "excluded_from_index": list(R.BANNED_FROM_DOCUMENT),
        "fusion": {
            "method": "reciprocal_rank_fusion",
            "k": settings.rrf_k,
            "dense_candidates": settings.dense_candidates,
            "lexical_candidates": settings.lexical_candidates,
            "metadata_boost_enabled": settings.metadata_boost_enabled,
            "metadata_boost_fraction_of_one_rrf_unit": settings.metadata_boost_fraction,
            "metadata_boost_absolute": round(
                R.metadata_boost_unit(settings.rrf_k, settings.metadata_boost_fraction), 6
            ),
        },
        "gate_thresholds": {
            "calibrated": settings.retrieval_gate_calibrated,
            "strong_min_cosine": settings.strong_min_cosine,
            "strong_min_margin": settings.strong_min_margin,
            "mixed_min_cosine": settings.mixed_min_cosine,
            "candidate_floor_cosine": settings.candidate_floor_cosine,
            "note": (
                "Calibrated for the embedding model named above. Changing the model "
                "invalidates these thresholds."
            ),
        },
    }
    with open(settings.retrieval_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nWrote retrieval artifacts to {settings.retrieval_dir}")
    print("Next: python -m backend.scripts.eval_retrieval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
