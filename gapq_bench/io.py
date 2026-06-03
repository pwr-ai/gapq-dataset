"""JSONL/JSON helpers, run-directory artifact names, packaged-prompt loading."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

# Fixed artifact names inside a run directory.
CONFIG = "config.json"
INPUT = "input.jsonl"
GENERATED = "generated.jsonl"
MATCHED = "matched.jsonl"
COVERAGE_INPUTS = "coverage_inputs.jsonl"
JUDGED = "judged.jsonl"
COVERAGE_JUDGED = "coverage_judged.jsonl"
SUMMARY = "summary.json"


def read_jsonl(path: Path | str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path | str, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path | str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_prompt(name: str) -> str:
    """Read a packaged prompt (gapq_bench/prompts/<name>.txt) verbatim."""
    return (files("gapq_bench") / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
