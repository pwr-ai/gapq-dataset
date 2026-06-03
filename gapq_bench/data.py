"""Load GapQ-PL from Hugging Face and build benchmark inputs.

Ported from the paper's research pipeline: strips the "Uzupełnienie" section
from `factual_state` (it contains the applicant's answers to the gold
supplementary questions — leaving it in would leak the targets), picks the
reference field per split, and drops unusable rows.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

DATASET_NAME = "AI-TAX/gapq-pl-dataset"

# Boundary of the "Uzupełnienie [i doprecyzowanie] [opisu] [wniosku]" section
# inside `factual_state`. Everything from the match onward is the applicant's
# reply to the office's supplementary call and must not be shown to the model.
UZUP_BOUNDARY_RE = re.compile(
    r"Uzupe[łl]nienie\s+(?:i\s+doprecyzowanie\s+)?(?:opisu\s+)?(?:wniosku)?",
    re.IGNORECASE,
)


def strip_uzupelnienie(text: str) -> str:
    """Remove the Uzupełnienie section from a factual_state string."""
    text = text or ""
    m = UZUP_BOUNDARY_RE.search(text)
    return text[: m.start()].rstrip() if m else text


def reference_field(split: str) -> str:
    """'verified' carries human-curated `questions`; 'extended' has empty
    `questions` by design, so its reference is the LLM-extracted
    `generated_questions`."""
    return "questions" if split == "verified" else "generated_questions"


def _reference_questions(rows) -> list[dict]:
    """Normalise the HF reference field to [{question, context}], robust to
    dict shapes that key on either `text` or `question`."""
    if not rows:
        return []
    out: list[dict] = []
    for r in rows:
        if isinstance(r, dict):
            t = (r.get("text") or r.get("question") or "").strip()
            if t:
                out.append({"question": t, "context": r.get("context") or ""})
    return out


def load_input_docs(
    dataset: str = DATASET_NAME,
    split: str = "verified",
    min_reference_questions: int = 1,
    limit: int | None = None,
) -> list[dict]:
    """One dict per benchmark doc, in dataset order:

    {doc_id, stan_faktyczny, tax_interpretation_questions, gold: [{question, context}]}
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    field = reference_field(split)

    docs: list[dict] = []
    n_no_ref = n_empty_stan = 0
    for row in ds:
        gold = _reference_questions(row.get(field))
        if len(gold) < min_reference_questions:
            n_no_ref += 1
            continue
        stan = strip_uzupelnienie(row.get("factual_state") or "")
        if not stan.strip():
            # Robustness guard: docs without a usable factual-state block
            # cannot be eval queries (the current HF release has none).
            n_empty_stan += 1
            continue
        docs.append(
            {
                "doc_id": str(row["doc_id"]),
                "stan_faktyczny": stan,
                "tax_interpretation_questions": (row.get("tax_interpretation_questions") or "").strip(),
                "gold": gold,
            }
        )

    logger.info(
        "[data] %s:%s → %d docs (dropped %d with <%d reference questions, "
        "%d with empty stan_faktyczny after Uzupełnienie strip)",
        dataset, split, len(docs), n_no_ref, min_reference_questions, n_empty_stan,
    )
    if limit is not None and limit > 0:
        docs = docs[:limit]
        logger.info("[data] --limit → keeping first %d docs", len(docs))
    return docs
