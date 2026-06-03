"""Judge stage: Topic & Area binary judges on matched pairs, Coverage judge
on top-K candidate sets.

Prompts, user-message formats and decision parsing are verbatim ports of the
paper's judges. Results are appended to JSONL as they complete, so an
interrupted run resumes without re-judging (or re-paying for) finished items;
failed calls are NOT written and get retried on the next invocation.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tqdm import tqdm

from gapq_bench.io import append_jsonl, load_prompt

logger = logging.getLogger(__name__)

PAIR_KEY = ("doc_id", "gold_question", "generated_question")
COVERAGE_KEY = ("doc_id", "gold_question")


class DimensionScore(BaseModel):
    decision: str = Field(description='„Tak" jeśli wymiar pasuje, „Nie" jeśli nie')
    reasoning: str = Field(description="Krótkie uzasadnienie decyzji")


class CoverageDecision(BaseModel):
    decision: str = Field(description='„Tak" lub „Nie"')


def _parse_decision(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith("tak") or '"tak"' in t or "'tak'" in t


def _done_keys(rows: list[dict], key_fields: tuple[str, ...]) -> set[tuple]:
    return {tuple(r[k] for k in key_fields) for r in rows}


async def judge_pairs(
    llm,
    pairs: list[dict],
    out_path: Path,
    existing: list[dict],
    concurrency: int = 20,
) -> int:
    """Judge Topic + Area for every matched pair not already in ``existing``.

    Appends each judged row (pair fields + topic/area bools + reasonings) to
    ``out_path``. Returns the number of failed pairs.
    """
    topic_prompt = load_prompt("judge_topic")
    area_prompt = load_prompt("judge_area")
    structured = llm.with_structured_output(DimensionScore)

    done = _done_keys(existing, PAIR_KEY)
    todo = [p for p in pairs if tuple(p[k] for k in PAIR_KEY) not in done]
    if done:
        logger.info("[judge] resuming: %d/%d pairs already judged", len(pairs) - len(todo), len(pairs))
    if not todo:
        return 0

    sem = asyncio.Semaphore(max(1, concurrency))
    pbar = tqdm(total=len(todo), desc="judge topic+area", unit="pair")
    n_failed = 0

    async def _dimension(system_prompt: str, user: str) -> DimensionScore:
        return await structured.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user)]
        )

    async def _one(pair: dict) -> None:
        nonlocal n_failed
        user = (
            f"Oryginalne pytanie:\n{pair['gold_question']}\n\n"
            f"Wygenerowane pytanie:\n{pair['generated_question']}\n\n"
            "Oceń binarnie (Tak/Nie) wskazany wymiar."
        )
        async with sem:
            try:
                # The two dimensions are independent inference passes.
                topic, area = await asyncio.gather(
                    _dimension(topic_prompt, user), _dimension(area_prompt, user)
                )
                row = dict(pair)
                row["topic"] = _parse_decision(topic.decision)
                row["area"] = _parse_decision(area.decision)
                row["topic_reasoning"] = topic.reasoning
                row["area_reasoning"] = area.reasoning
                append_jsonl(out_path, row)
            except Exception as exc:
                n_failed += 1
                logger.warning("[judge] pair failed (%.60s…): %r", pair["gold_question"], exc)
            finally:
                pbar.update(1)

    await asyncio.gather(*(_one(p) for p in todo))
    pbar.close()
    if n_failed:
        logger.warning("[judge] %d pairs failed — rerun the judge stage to retry them", n_failed)
    return n_failed


async def judge_coverage(
    llm,
    items: list[dict],
    out_path: Path,
    existing: list[dict],
    concurrency: int = 20,
) -> int:
    """Judge coverage for every gold question not already in ``existing``.

    Items with no candidates count as not covered without an LLM call,
    mirroring the original scorer. Returns the number of failed items.
    """
    system_prompt = load_prompt("llm_matcher")
    structured = llm.with_structured_output(CoverageDecision)

    done = _done_keys(existing, COVERAGE_KEY)
    todo = [it for it in items if tuple(it[k] for k in COVERAGE_KEY) not in done]
    if done:
        logger.info(
            "[coverage] resuming: %d/%d gold questions already judged",
            len(items) - len(todo), len(items),
        )
    if not todo:
        return 0

    sem = asyncio.Semaphore(max(1, concurrency))
    pbar = tqdm(total=len(todo), desc="judge coverage", unit="gold")
    n_failed = 0

    async def _one(item: dict) -> None:
        nonlocal n_failed
        candidates = item["top_candidates"]
        if not candidates:
            append_jsonl(out_path, {**item, "covered": False})
            pbar.update(1)
            return
        cands_txt = "\n".join(f"- {c}" for c in candidates)
        user = (
            f"PYTANIE_ORYGINALNE:\n{item['gold_question']}\n\n"
            f"PYTANIA_PRZYPASOWANE:\n{cands_txt}"
        )
        async with sem:
            try:
                result = await structured.ainvoke(
                    [SystemMessage(content=system_prompt), HumanMessage(content=user)]
                )
                covered = result.decision.strip().lower().startswith("tak")
                append_jsonl(out_path, {**item, "covered": covered})
            except Exception as exc:
                n_failed += 1
                logger.warning("[coverage] item failed (%.60s…): %r", item["gold_question"], exc)
            finally:
                pbar.update(1)

    await asyncio.gather(*(_one(it) for it in todo))
    pbar.close()
    if n_failed:
        logger.warning("[coverage] %d items failed — rerun the judge stage to retry them", n_failed)
    return n_failed
