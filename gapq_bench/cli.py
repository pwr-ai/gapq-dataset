"""gapq-bench CLI — run the GapQ-PL generation benchmark end-to-end or stage by stage.

    gapq-bench run --model openai:gpt-5.2 --limit 5
    gapq-bench generate | match | judge | report   # individual stages

Stages communicate through a run directory (default: runs/<UTC timestamp>;
stage commands without --run-dir pick the most recent one).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

from gapq_bench import __version__, io
from gapq_bench.data import DATASET_NAME, load_input_docs
from gapq_bench.match import DEFAULT_TOP_K, EMBED_MODEL_NAME

logger = logging.getLogger("gapq_bench")


# ---------------------------------------------------------------------------
# helpers


def _init_chat_model(model: str, temperature: float | None):
    from langchain.chat_models import init_chat_model

    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        return init_chat_model(model, **kwargs)
    except ImportError as exc:
        provider = model.split(":", 1)[0]
        raise SystemExit(
            f"Missing integration package for provider '{provider}' ({exc}).\n"
            f"Install it, e.g.: pip install 'gapq-bench[{provider}]'"
        ) from exc


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"gapq-bench": __version__, "python": platform.python_version()}
    for pkg in ("langchain", "datasets", "sentence-transformers"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _update_config(run_dir: Path, **kwargs) -> None:
    path = run_dir / io.CONFIG
    cfg = io.read_json(path) if path.exists() else {}
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    cfg["versions"] = _versions()
    io.write_json(path, cfg)


def _resolve_run_dir(args, create: bool) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    runs = Path("runs")
    if create:
        return runs / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidates = sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []
    if not candidates:
        raise SystemExit("No run directory found — pass --run-dir or start with `gapq-bench generate`.")
    return candidates[-1]


def _require(run_dir: Path, name: str) -> list[dict]:
    path = run_dir / name
    if not path.exists():
        raise SystemExit(f"Missing {path} — run the earlier stages first.")
    return io.read_jsonl(path)


# ---------------------------------------------------------------------------
# stages


def stage_generate(args, run_dir: Path) -> None:
    from gapq_bench.generate import generate_questions

    input_path = run_dir / io.INPUT
    gen_path = run_dir / io.GENERATED

    if input_path.exists() and not args.force:
        docs = io.read_jsonl(input_path)
        logger.info("[generate] using existing %s (%d docs)", input_path, len(docs))
    else:
        docs = load_input_docs(args.dataset, args.split, args.min_reference_questions, args.limit)
        io.write_jsonl(input_path, docs)
    _update_config(
        run_dir,
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        min_reference_questions=args.min_reference_questions,
        model=args.model,
        temperature=args.temperature,
        concurrency=args.concurrency,
    )

    if gen_path.exists():
        if args.force:
            gen_path.unlink()
        elif args.resume:
            done = {r["doc_id"] for r in io.read_jsonl(gen_path)}
            docs = [d for d in docs if d["doc_id"] not in done]
            logger.info("[generate] resuming: %d docs left", len(docs))
        else:
            logger.info(
                "[generate] %s exists — skipping (--force regenerates, --resume fills gaps)",
                gen_path,
            )
            return
    if not docs:
        logger.info("[generate] nothing to do")
        return

    llm = _init_chat_model(args.model, args.temperature)

    def _on_doc_done(doc_id: str, rows: list[dict]) -> None:
        for row in rows:
            io.append_jsonl(gen_path, row)

    generate_questions(llm, docs, _on_doc_done, concurrency=args.concurrency)


def stage_match(args, run_dir: Path) -> None:
    from gapq_bench.match import build_matches, load_embedder

    matched_path = run_dir / io.MATCHED
    coverage_path = run_dir / io.COVERAGE_INPUTS
    if matched_path.exists() and coverage_path.exists() and not args.force:
        logger.info("[match] %s exists — skipping (--force re-runs)", matched_path)
        return

    docs = _require(run_dir, io.INPUT)
    generated = _require(run_dir, io.GENERATED)
    _update_config(run_dir, embed_model=args.embed_model, top_k=args.top_k)

    embedder = load_embedder(args.embed_model)
    matched, coverage = build_matches(docs, generated, embedder, top_k=args.top_k)
    io.write_jsonl(matched_path, matched)
    io.write_jsonl(coverage_path, coverage)


def stage_judge(args, run_dir: Path) -> None:
    from gapq_bench.judge import judge_coverage, judge_pairs

    judged_path = run_dir / io.JUDGED
    cov_judged_path = run_dir / io.COVERAGE_JUDGED
    if args.force:
        judged_path.unlink(missing_ok=True)
        cov_judged_path.unlink(missing_ok=True)

    pairs = _require(run_dir, io.MATCHED)
    coverage_items = _require(run_dir, io.COVERAGE_INPUTS)
    _update_config(
        run_dir,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_concurrency=args.judge_concurrency,
    )

    llm = _init_chat_model(args.judge_model, args.judge_temperature)
    existing_pairs = io.read_jsonl(judged_path) if judged_path.exists() else []
    existing_cov = io.read_jsonl(cov_judged_path) if cov_judged_path.exists() else []
    asyncio.run(
        judge_pairs(llm, pairs, judged_path, existing_pairs, concurrency=args.judge_concurrency)
    )
    asyncio.run(
        judge_coverage(
            llm, coverage_items, cov_judged_path, existing_cov, concurrency=args.judge_concurrency
        )
    )


def stage_report(args, run_dir: Path) -> None:
    from gapq_bench.report import build_summary, format_summary

    docs = _require(run_dir, io.INPUT)
    generated = _require(run_dir, io.GENERATED)
    matched = _require(run_dir, io.MATCHED)
    judged = _require(run_dir, io.JUDGED)
    coverage = _require(run_dir, io.COVERAGE_JUDGED)

    summary = build_summary(docs, generated, matched, judged, coverage)
    io.write_json(run_dir / io.SUMMARY, summary)
    config_path = run_dir / io.CONFIG
    config = io.read_json(config_path) if config_path.exists() else {}
    print(format_summary(summary, config))
    print(f"\nFull summary: {run_dir / io.SUMMARY}")


# ---------------------------------------------------------------------------
# argument parsing


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", default=None, help="Run directory (default: runs/<timestamp> for generate/run, latest otherwise)")
    p.add_argument("--force", action="store_true", help="Re-run the stage even if its output exists")


def _add_data(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", default=DATASET_NAME)
    p.add_argument("--split", choices=["verified", "extended"], default="verified")
    p.add_argument("--limit", type=int, default=None, help="Only the first N docs (smoke tests)")
    p.add_argument("--min-reference-questions", type=int, default=1)


def _add_generate(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help='Generator for init_chat_model, e.g. "openai:gpt-5.2", "anthropic:claude-opus-4-8"')
    p.add_argument("--temperature", type=float, default=1.0, help="Generator temperature (paper: 1.0)")
    p.add_argument("--concurrency", type=int, default=25)
    p.add_argument("--resume", action="store_true", help="Only generate for docs missing from generated.jsonl")


def _add_match(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embed-model", default=EMBED_MODEL_NAME)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Coverage candidates per gold question (paper: 20)")


def _add_judge(p: argparse.ArgumentParser) -> None:
    p.add_argument("--judge-model", default="openai:gpt-5-mini", help="Judge for init_chat_model (paper: gpt-5-mini)")
    p.add_argument("--judge-temperature", type=float, default=None, help="Judge temperature (default: provider default; gpt-5 family rejects values ≠ 1.0)")
    p.add_argument("--judge-concurrency", type=int, default=20)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fsspec", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="gapq-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"gapq-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="generate → match → judge → report")
    for add in (_add_common, _add_data, _add_generate, _add_match, _add_judge):
        add(p_run)

    p_gen = sub.add_parser("generate", help="LLM generates supplementary questions")
    for add in (_add_common, _add_data, _add_generate):
        add(p_gen)

    p_match = sub.add_parser("match", help="embed + match generated questions to gold")
    for add in (_add_common, _add_match):
        add(p_match)

    p_judge = sub.add_parser("judge", help="LLM-as-judge: Topic, Area, Coverage")
    for add in (_add_common, _add_judge):
        add(p_judge)

    p_report = sub.add_parser("report", help="aggregate metrics into summary.json")
    _add_common(p_report)

    args = parser.parse_args(argv)

    create = args.command in ("run", "generate")
    run_dir = _resolve_run_dir(args, create=create)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[gapq-bench] run dir: %s", run_dir)

    if args.command == "run":
        stage_generate(args, run_dir)
        stage_match(args, run_dir)
        stage_judge(args, run_dir)
        stage_report(args, run_dir)
    elif args.command == "generate":
        stage_generate(args, run_dir)
    elif args.command == "match":
        stage_match(args, run_dir)
    elif args.command == "judge":
        stage_judge(args, run_dir)
    elif args.command == "report":
        stage_report(args, run_dir)


if __name__ == "__main__":
    main()
