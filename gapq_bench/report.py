"""Aggregate run artifacts into the paper's benchmark metrics.

Headline metrics (cf. the zero-shot table in the paper): number of predicted
questions, over-generation ratio |Qp|/|Qg|, Topic %, Area %, Coverage %, and
mean embedding similarity of the top-1 matches. Rates are means over pairs /
gold questions across the whole split, plus a per-doc breakdown.
"""

from __future__ import annotations


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def build_summary(
    docs: list[dict],
    generated_rows: list[dict],
    matched_rows: list[dict],
    judged_rows: list[dict],
    coverage_rows: list[dict],
) -> dict:
    n_gold = sum(len(d["gold"]) for d in docs)
    n_predicted = len(generated_rows)

    per_doc: dict[str, dict] = {}
    for d in docs:
        per_doc[d["doc_id"]] = {
            "doc_id": d["doc_id"],
            "n_gold": len(d["gold"]),
            "n_generated": 0,
            "avg_embedding_similarity": None,
            "topic_match_rate": None,
            "area_match_rate": None,
            "coverage_rate": None,
        }
    for r in generated_rows:
        if r["doc_id"] in per_doc:
            per_doc[r["doc_id"]]["n_generated"] += 1

    def _doc_rates(rows: list[dict], value_key: str, out_key: str) -> None:
        by_doc: dict[str, list] = {}
        for r in rows:
            by_doc.setdefault(r["doc_id"], []).append(r[value_key])
        for doc_id, vals in by_doc.items():
            if doc_id in per_doc:
                per_doc[doc_id][out_key] = round(_mean(vals), 3)

    _doc_rates(matched_rows, "embedding_similarity", "avg_embedding_similarity")
    _doc_rates(judged_rows, "topic", "topic_match_rate")
    _doc_rates(judged_rows, "area", "area_match_rate")
    _doc_rates(coverage_rows, "covered", "coverage_rate")

    return {
        "n_docs": len(docs),
        "n_gold": n_gold,
        "n_predicted": n_predicted,
        "overgeneration": round(n_predicted / n_gold, 2) if n_gold else 0.0,
        "n_judged_pairs": len(judged_rows),
        "n_coverage_items": len(coverage_rows),
        "avg_embedding_similarity": round(_mean(r["embedding_similarity"] for r in matched_rows), 3),
        "topic_match_rate": round(_mean(r["topic"] for r in judged_rows), 3),
        "area_match_rate": round(_mean(r["area"] for r in judged_rows), 3),
        "coverage_rate": round(_mean(r["covered"] for r in coverage_rows), 3),
        "per_doc": list(per_doc.values()),
    }


def format_summary(summary: dict, config: dict | None = None) -> str:
    config = config or {}
    lines = [
        "GapQ-PL benchmark"
        + (f" — split={config['split']}" if config.get("split") else "")
        + (f"  model={config['model']}" if config.get("model") else "")
        + (f"  judge={config['judge_model']}" if config.get("judge_model") else ""),
        (
            f"docs {summary['n_docs']}   gold |Qg| {summary['n_gold']}   "
            f"predicted |Qp| {summary['n_predicted']}   "
            f"|Qp|/|Qg| {summary['overgeneration']:.2f}"
        ),
        (
            f"Topic {summary['topic_match_rate']:.1%}   "
            f"Area {summary['area_match_rate']:.1%}   "
            f"Coverage {summary['coverage_rate']:.1%}   "
            f"Emb. sim {summary['avg_embedding_similarity']:.3f}"
        ),
    ]
    return "\n".join(lines)
