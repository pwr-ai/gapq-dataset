# GapQ-PL

**Question Generation for Gaps in Polish Tax Interpretations**

Companion repository for the paper *GapQ: Question Generation for Gaps in
Polish Tax Interpretations* (in review, NeurIPS 2026 Datasets & Benchmarks
Track). The dataset itself lives on Hugging Face:

> https://huggingface.co/datasets/AI-TAX/gapq-pl-dataset

## What is GapQ-PL?

GapQ-PL is a curated Polish-language dataset of *supplementary
clarification questions* that the Polish tax office (KIS — *Krajowa
Informacja Skarbowa*) issues to applicants when an individual
tax-interpretation request is missing factual details. The benchmark
task is: given the applicant's factual state and the original
interpretation questions, generate the supplementary clarification
questions the office would emit.

## Splits

| split      | docs   | curation                                                |
|------------|-------:|---------------------------------------------------------|
| `verified` |   500  | row-by-row human revision (gold)                        |
| `extended` | 1,897  | LLM-extracted, deterministic post-checks, no human pass |

Both splits are drawn from the same pool of $2{,}353$ manually-screened
candidate documents whose supplementary section is wholly in Q&A format.

## Quick start (dataset only)

```python
from datasets import load_dataset
ds = load_dataset("AI-TAX/gapq-pl-dataset")
ds["verified"][0]
```

## Running the benchmark

This repo ships `gapq-bench`, a minimal reference harness that reproduces
the paper's evaluation pipeline against the official Hugging Face dataset:

1. **generate** — for each document, an LLM generates the office's
   supplementary questions from `factual_state` (+ the taxpayer's
   interpretation questions), using the paper's verbatim Polish zero-shot
   prompt and structured output. The `Uzupełnienie` section is stripped
   from `factual_state` first — it quotes the gold questions.
2. **match** — gold and generated questions are embedded with
   [`sdadas/st-polish-paraphrase-from-distilroberta`](https://huggingface.co/sdadas/st-polish-paraphrase-from-distilroberta);
   each gold question greedily takes its most-similar generated question
   (many-to-one, no threshold), and its top-20 candidates are kept for the
   coverage judge.
3. **judge** — an LLM judge (paper: `gpt-5-mini`) scores each matched pair
   on two binary dimensions (**Topic**: same subject-matter issue;
   **Area**: same tax/legal bucket) and, per gold question, whether the
   top-20 candidates jointly **cover** all of its information requirements.
4. **report** — aggregates into the paper's headline metrics.

### Install

```bash
pip install -e '.[openai]'        # or [anthropic], [google], [all]
# or: uv pip install -e '.[openai]'
```

Bring your own API key via the provider's environment variable
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …; see `.env.example`). Models are
addressed with LangChain `init_chat_model` strings — any provider with a
LangChain integration and structured-output support works.

First run downloads the dataset (~30 MB), the Polish paraphrase encoder
(~500 MB), and PyTorch as a dependency of `sentence-transformers`.

### Usage

```bash
# cheap smoke test: 3 documents end-to-end
gapq-bench run --model openai:gpt-5-mini --limit 3 --run-dir runs/smoke

# full verified-split benchmark (paper setting)
gapq-bench run --model openai:gpt-5.2 --judge-model openai:gpt-5-mini --split verified

# stages can also be run separately, sharing a run directory:
gapq-bench generate --model anthropic:claude-opus-4-8 --run-dir runs/claude
gapq-bench match    --run-dir runs/claude
gapq-bench judge    --run-dir runs/claude
gapq-bench report   --run-dir runs/claude
```

Each run directory holds the intermediate artifacts
(`input.jsonl` → `generated.jsonl` → `matched.jsonl` / `coverage_inputs.jsonl`
→ `judged.jsonl` / `coverage_judged.jsonl` → `summary.json`) plus a frozen
`config.json`. Judge results are appended incrementally — an interrupted run
resumes for free; a full `verified` run is ≈21k judge calls. Generation
re-runs are opt-in via `--resume` (it fills in missing docs only) or
`--force` (regenerate all).

### Metrics & paper reference numbers

| metric | meaning |
|---|---|
| \|Qp\|/\|Qg\| | over-generation: predicted vs gold question count |
| Topic | % of matched pairs judged the same subject-matter issue |
| Area | % of matched pairs judged the same tax/legal bucket |
| Coverage | % of gold questions whose top-20 candidates jointly cover all information requirements |
| Emb. sim | mean cosine similarity of top-1 matches |

Zero-shot results on `verified` from the paper (judge `gpt-5-mini`), as a
sanity target for reproduction:

| model | \|Qp\|/\|Qg\| | Topic | Area | Coverage | Emb. sim |
|---|---:|---:|---:|---:|---:|
| gpt-5.2 | 2.20× | 49.6% | 89.7% | 53.3% | 0.705 |
| gpt-5-mini | 1.49× | 34.8% | 83.0% | 38.5% | 0.658 |
| gpt-oss-120b | 1.16× | 26.8% | 72.6% | 21.1% | 0.619 |
| deepseek-v3-0324 | 0.71× | 23.0% | 75.8% | 10.6% | 0.631 |

**Reproducibility notes.** Generation runs at temperature 1.0 (paper
setting), so numbers vary run to run; the paper's K=5/L=800 noise check
puts run-to-run variation at ≈1–4 pp. The paper's judge calls went through
the OpenAI Batch API at temperature 1.0 (the gpt-5 family rejects other
values); the harness leaves judge temperature at the provider default
unless `--judge-temperature` is set.

## Files in this repo

- `README.md` — this file.
- `croissant.json` — Croissant 1.1 metadata, fetched from the Hugging
  Face Croissant endpoint
  (`https://huggingface.co/api/datasets/AI-TAX/gapq-pl-dataset/croissant`).
- `pyproject.toml`, `gapq_bench/` — the `gapq-bench` benchmark harness
  (CLI: `gapq-bench`). The paper's Polish generator/judge prompts ship
  verbatim in `gapq_bench/prompts/`.
- `LICENSE` — MIT, covering the code in this repository.

## License

The **code** in this repository is released under the **MIT License**
(see `LICENSE`). The **dataset** is released under **CC BY 4.0**, matching
the Hugging Face release.

## Citation

```bibtex
@inproceedings{gapq2026,
  title  = {GapQ: Question Generation for Gaps in Polish Tax Interpretations},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at NeurIPS 2026 Datasets \& Benchmarks Track}
}
```
