import json
from pathlib import Path

import pandas as pd

from compare_runs import summarize_run


def test_summarize_run_includes_theme_details():
    run = {
        "timestamp": "2026-11-02T21:23:54",
        "model": "openai/gpt-4o-mini",
        "temperature": 0,
        "result": {
            "themes": [
                {"topic": "Bus fares and accessibility", "topic_id": "A"},
                {"topic": "Service reliability", "topic_id": "B"},
            ]
        },
    }

    summary = summarize_run(run, "run_20261102_212354.json")

    assert summary["theme_count"] == 2
    assert summary["topics"] == [
        "Bus fares and accessibility",
        "Service reliability",
    ]
    assert summary["mentions_fares"] is True


def test_input_and_gold_align():
    repo_root = Path(__file__).resolve().parents[1]
    input_df = pd.read_csv(repo_root / "data" / "golden" / "themefinder_input.csv")
    gold_df = pd.read_csv(repo_root / "data" / "golden" / "initial_annotation_sample_reviewed.csv")

    joined = input_df.merge(gold_df, left_on="response_id", right_on="ResponseID", how="inner")

    assert len(joined) == len(gold_df)
    assert set(input_df["response_id"]) == set(gold_df["ResponseID"])
    assert all(str(comment).strip() in str(response).strip() for comment, response in zip(joined["Comment"], joined["response"]))
