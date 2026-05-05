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

## Quick start

```python
from datasets import load_dataset
ds = load_dataset("AI-TAX/gapq-pl-dataset")
ds["verified"][0]
```

## Files in this repo

- `README.md` — this file.
- `croissant.json` — Croissant 1.1 metadata, fetched from the Hugging
  Face Croissant endpoint
  (`https://huggingface.co/api/datasets/AI-TAX/gapq-pl-dataset/croissant`).

## License

The dataset is released under **CC BY 4.0**, matching the Hugging Face
release.

## Citation

```bibtex
@inproceedings{gapq2026,
  title  = {GapQ: Question Generation for Gaps in Polish Tax Interpretations},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review at NeurIPS 2026 Datasets \& Benchmarks Track}
}
```
