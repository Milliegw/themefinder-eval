# ThemeFinder evaluation

This project evaluates how well the ThemeFinder theme-classification pipeline performs on public consultation responses, and whether a local LLM can be used as a pre-screening layer before human annotation.

## Project goal

We want to:

- build a usable golden dataset from real consultation comments,
- remove clearly low-quality or empty responses before annotation,
- run a local LLM to flag likely unusable rows,
- manually review the flagged rows,
- evaluate ThemeFinder against the final reviewed gold set.

## Key files and what they do

### Raw data

- `data/council-plan-consultation-19-report-annex.csv`
  - Original council consultation CSV.
  - This is the source dataset used to build the annotation sample.

### Data preparation

- `explore_data.py`
  - Cleans the raw consultation data.
  - Drops empty or whitespace-only comments.
  - Flags obvious low-quality comments such as “no idea”, “not sure”, or “nothing to add”.
  - Builds a balanced sample across outcome groups.
  - Saves the annotation set to `data/golden/initial_annotation_sample.csv`.

- `data/golden/initial_annotation_sample.csv`
  - The first balanced annotation sample created from the raw data.
  - This is the file reviewed manually or by the local model before final gold-labeling.

### LLM review

- `run_local_llm_review.py`
  - Reads the annotation sample.
  - Sends each comment to a local Ollama model.
  - Labels each row as either `keep` or `flag_for_review`.
  - Saves the result in `data/golden/initial_annotation_sample_llm.csv`.

- `data/golden/initial_annotation_sample_llm.csv`
  - Full LLM review output.
  - Includes the original `Comment`, `Outcome`, `Question`, plus:
    - `llm_label`
    - `llm_reason`
  - This file should be treated as the audit trail for the LLM pass.

### Final reviewed set

- `data/golden/initial_annotation_sample_reviewed.csv`
  - Final cleaned review set after removing all rows marked `flag_for_review`.
  - This is the version to use downstream for human labeling or evaluation.
  - It keeps the useful rows only and excludes obviously weak content.

### Evaluation and comparison

- `compare_runs.py`
  - Compares ThemeFinder outputs across runs.

- `run_themefinder.py`
  - Runs the ThemeFinder pipeline locally.

- `src/metrics.py`
  - Evaluation metrics such as precision, recall, F1, and bootstrap confidence intervals.

- `test/`
  - Regression and metric tests.

## Recommended workflow

1. Start with the raw consultation CSV.
2. Run `python explore_data.py`.
   - This cleans the source data and creates the initial balanced sample.
3. Run the local LLM review:
   - `python run_local_llm_review.py`
   - This creates `data/golden/initial_annotation_sample_llm.csv`.
4. Manually review the rows flagged `flag_for_review`.
5. Remove all flagged rows from the final dataset.
6. Keep a filtered file containing only `llm_label == "keep"`.
7. Use this reviewed file as the basis for the final gold dataset and downstream evaluation.

## Best practice for filtering out flagged rows

The cleanest approach is to keep the original LLM output as a record and create a second filtered file.

Example:

```python
import pandas as pd

path = "data/golden/initial_annotation_sample_llm.csv"
df = pd.read_csv(path)
filtered = df[df["llm_label"] == "keep"]
filtered.to_csv("data/golden/initial_annotation_sample_reviewed.csv", index=False)
```

This keeps the raw LLM file for auditing while creating a clean downstream dataset for labeling and evaluation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

ThemeFinder runs locally through Ollama's OpenAI-compatible endpoint; for example:

```bash
ollama pull llama3.1
```

## Current status

- [x] Raw consultation data identified and loaded correctly
- [x] Empty/no-comment rows removed
- [x] Balanced sample created for annotation
- [x] Local LLM review pipeline built and run
- [x] Flagged rows reviewed and removed from the final cleaned set
- [ ] Final hand-labelled gold set produced
- [ ] ThemeFinder evaluated against the gold set
- [ ] LLM-as-a-judge comparison completed

## Notes

- LLM review is a useful filtering layer, not a final source of truth.
- Human review remains important for rows marked `flag_for_review`.
- The final reviewed dataset should remain smaller, cleaner, and more reliable for evaluation work.
