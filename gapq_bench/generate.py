"""Generation stage: stan faktyczny (+ taxpayer questions) → supplementary questions.

The pydantic schema and the user-message assembly are verbatim ports of the
paper's research code — the Polish field descriptions enter the structured-
output JSON schema and are load-bearing.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tqdm import tqdm

from gapq_bench.io import load_prompt

logger = logging.getLogger(__name__)


class SupplementaryQuestion(BaseModel):
    question: str = Field(description="Treść pytania uzupełniającego")
    context: str = Field(
        default="",
        description=(
            "Dla pytania głównego — pusty łańcuch \"\". "
            "Dla podpytania — dosłowna treść pytania głównego "
            "(znak po znaku identyczna we wszystkich podpytaniach jednej grupy)."
        ),
    )


class SupplementaryQuestionList(BaseModel):
    questions: list[SupplementaryQuestion] = Field(description="Lista pytań uzupełniających")


def build_messages(system_prompt: str, stan: str, pytania: str) -> list[BaseMessage]:
    """Exact user-message assembly from the paper's zero-shot generator."""
    human_parts: list[str] = [f"Stan faktyczny:\n{stan}"]
    if pytania:
        human_parts.append("")
        human_parts.append(f"Pytania podatnika:\n{pytania}")
    human_parts.append("")
    human_parts.append("Wygeneruj pytania uzupełniające.")
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(human_parts)),
    ]


def generate_questions(
    llm,
    docs: list[dict],
    on_doc_done: Callable[[str, list[dict]], None],
    concurrency: int = 25,
) -> int:
    """Generate questions for every doc; call ``on_doc_done(doc_id, rows)`` as
    each doc completes (rows = [{doc_id, question, context}, ...]).

    Failed docs are logged and skipped (no callback), so a ``--resume`` rerun
    retries them. Returns the number of failed docs.
    """
    system_prompt = load_prompt("generator_general")
    structured_llm = llm.with_structured_output(SupplementaryQuestionList)

    def _one(doc: dict) -> list[dict]:
        messages = build_messages(
            system_prompt, doc["stan_faktyczny"], doc["tax_interpretation_questions"]
        )
        result = structured_llm.invoke(messages)
        if result is None:
            raise ValueError("structured output returned None")
        return [
            {"doc_id": doc["doc_id"], "question": q.question, "context": q.context}
            for q in result.questions
        ]

    n_failed = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = {ex.submit(_one, doc): doc["doc_id"] for doc in docs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="generate", unit="doc"):
            doc_id = futures[fut]
            try:
                rows = fut.result()
            except Exception:
                n_failed += 1
                logger.exception("[generate] doc %s failed", doc_id)
                continue
            on_doc_done(doc_id, rows)
    if n_failed:
        logger.warning(
            "[generate] %d/%d docs failed — rerun with --resume to retry them",
            n_failed, len(docs),
        )
    return n_failed
