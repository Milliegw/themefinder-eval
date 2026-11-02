import json

from runs.compare_runs import summarize_run


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
