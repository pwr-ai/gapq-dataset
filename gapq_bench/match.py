"""Matching stage: embed gold + generated questions per doc and produce

1. gold-anchored top-1 pairs for the Topic/Area judges, and
2. top-K candidate sets per gold question for the Coverage judge.

Numeric logic is a verbatim port of the paper's matcher: normalized
embeddings (dot product = cosine), greedy many-to-one argmax per gold
question, NO similarity threshold.
"""

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "sdadas/st-polish-paraphrase-from-distilroberta"
DEFAULT_TOP_K = 20


def load_embedder(name: str = EMBED_MODEL_NAME):
    # Lazy import: sentence-transformers pulls in torch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def _embed(embedder, texts: list[str]) -> np.ndarray:
    return np.asarray(
        embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    )


def build_matches(
    docs: list[dict],
    generated_rows: list[dict],
    embedder,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[dict], list[dict]]:
    """Return ``(matched_rows, coverage_rows)``.

    matched_rows  — one per gold question in docs with ≥1 generated question:
        {doc_id, gold_question, generated_question, embedding_similarity}
    coverage_rows — one per gold question in EVERY doc (docs without generated
        questions get empty candidate lists and later count as not covered,
        mirroring the original scorer):
        {doc_id, gold_question, top_candidates, top_similarity}
    """
    gen_by_doc: dict[str, list[str]] = {}
    for r in generated_rows:
        gen_by_doc.setdefault(str(r["doc_id"]), []).append(r["question"])

    matched: list[dict] = []
    coverage: list[dict] = []
    for doc in tqdm(docs, desc="match", unit="doc"):
        doc_id = doc["doc_id"]
        gold = [g["question"] for g in doc["gold"]]
        gen = gen_by_doc.get(doc_id, [])
        if not gold:
            continue
        if not gen:
            for gold_q in gold:
                coverage.append(
                    {
                        "doc_id": doc_id,
                        "gold_question": gold_q,
                        "top_candidates": [],
                        "top_similarity": 0.0,
                    }
                )
            continue

        gold_emb = _embed(embedder, gold)
        gen_emb = _embed(embedder, gen)
        sims = gold_emb @ gen_emb.T
        k = min(top_k, len(gen))
        for i, gold_q in enumerate(gold):
            row = sims[i]
            # Top-1 greedy assignment (many-to-one, no threshold).
            matched.append(
                {
                    "doc_id": doc_id,
                    "gold_question": gold_q,
                    "generated_question": gen[int(np.argmax(row))],
                    "embedding_similarity": round(float(row.max()), 3),
                }
            )
            # Ranked top-k candidates for the coverage judge.
            idx = np.argpartition(-row, k - 1)[:k]
            ranked = sorted(
                ((int(j), float(row[j])) for j in idx), key=lambda x: x[1], reverse=True
            )
            coverage.append(
                {
                    "doc_id": doc_id,
                    "gold_question": gold_q,
                    "top_candidates": [gen[j] for j, _ in ranked],
                    "top_similarity": round(ranked[0][1], 3) if ranked else 0.0,
                }
            )

    n_docs = len({r["doc_id"] for r in matched})
    avg = sum(r["embedding_similarity"] for r in matched) / len(matched) if matched else 0.0
    logger.info(
        "[match] %d top-1 pairs across %d docs (avg similarity %.3f); "
        "%d coverage rows (top_k=%d)",
        len(matched), n_docs, avg, len(coverage), top_k,
    )
    return matched, coverage
