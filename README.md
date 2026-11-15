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

- `run_themefinder_test.py`
  - Runs the project-specific ThemeFinder pipeline against the cleaned comment-only input in `data/golden/themefinder_input.csv`.
  - Saves the JSON output into `runs/`.

- `compare_themefinder_gold.py`
  - Reads the most recent ThemeFinder JSON output and compares it against the reviewed gold set.
  - Aligns rows by `response_id` and maps output labels onto a stable human-readable theme codebook before scoring.
  - Prints the per-theme precision/recall/F1 values and writes a Markdown report into `runs/*.md`.

- `eval_pydantic.py`
  - Offline evaluation wrapper built with `pydantic-evals`.
  - Uses the reviewed gold data and saved ThemeFinder output to compute per-theme evaluation metrics.

- `compare_runs.py`
  - Compares ThemeFinder outputs across runs.

- `run_themefinder.py`
  - Earlier prototype or example ThemeFinder run.

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
7. Use this reviewed file as the basis for the final gold dataset.
8. Build the cleaned comment-only ThemeFinder input in `data/golden/themefinder_input.csv`.
   - This input uses only the comment text and aligns rows by `response_id` to the reviewed gold dataset.
9. Run `python run_themefinder_test.py`.
10. Run `python compare_themefinder_gold.py`.
    - This compares the ThemeFinder output against the gold set using `response_id` alignment and a stable codebook mapping.
11. Optionally run the offline `pydantic-evals` evaluation wrapper via `python eval_pydantic.py`.
12. Save and review the Markdown report written to `runs/*.md`.

## How to interpret the metrics

The comparison script reports per-theme precision, recall, and F1 for the ThemeFinder topic IDs.

- A good score is typically materially above 0.0 F1 for a theme that is well-supported in the gold data, and the value will vary by how common the theme is and how clearly the model separates it from nearby classes.
- Very low or zero scores often mean one of two things:
  - the theme is rare in the gold data,
  - the model is clustering comments differently from the human coding.
- In this mock setup, the key thing is to keep the gold data and model labels in the same taxonomy before comparing them.

## Results and findings

The comparison script produces a per-theme summary showing precision, recall, F1, and a bootstrap 95% interval for F1. This makes it easier to see whether a weak theme score is a genuine modelling failure or simply a rare theme with low support in the reviewed data.

The current evaluation pipeline compares ThemeFinder output to the reviewed human labels by aligning on `response_id` and mapping the model’s labels into the project’s stable theme codebook. This avoids run-specific Topic A–H drift and keeps the benchmark anchored to the actual consultation taxonomy used in the gold dataset.

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
- [x] Gold review set produced
- [x] Clean comment-only ThemeFinder input prepared and aligned by `response_id`
- [x] ThemeFinder run executed and JSON output saved in `runs/`
- [x] ThemeFinder output compared against the gold set using a stable theme codebook
- [x] Offline `pydantic-evals` evaluation wrapper added
- [ ] Refine theme clustering or taxonomy mapping if a more human-friendly benchmark is needed
- [ ] LLM-as-a-judge comparison completed

## Notes

- LLM review is a useful filtering layer, not a final source of truth.
- Human review remains important for rows marked `flag_for_review`.
- The final reviewed dataset should remain smaller, cleaner, and more reliable for evaluation work.
