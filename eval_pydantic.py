"""Offline ThemeFinder evaluation using pydantic-evals.

This script builds a dataset directly from the reviewed gold CSV, scores the saved
ThemeFinder run without calling an LLM, and reports per-theme precision/recall/F1.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from compare_themefinder_gold import GOLD_TO_THEMEFINDER_ID, latest_themefinder_json
from src.metrics import f1, precision, recall


THEMEFINDER_ID_TO_GOLD = {
    value: key for key, value in GOLD_TO_THEMEFINDER_ID.items() if value is not None
}


def normalize_text(value: str | None) -> str:
    """Collapse whitespace so comment matching is stable across CSV and JSON inputs."""
    return " ".join(str(value or "").strip().split())


def extract_comment_from_response_text(response_text: str | None) -> str:
    """Pull the raw comment out of the ThemeFinder response payload."""
    text = str(response_text or "")
    match = re.search(r"Comment:\s*(.*)$", text, flags=re.DOTALL)
    if match:
        return normalize_text(match.group(1))
    return normalize_text(text)


def load_gold_cases(csv_path: str | Path) -> tuple[list[Case[str, set[str], None]], list[str]]:
    """Build dataset cases from the reviewed gold CSV.

    The saved ThemeFinder JSON does not reliably line up by numeric response_id,
    so we match on comment text instead; the case names still use the ResponseID.
    """
    df = pd.read_csv(csv_path)
    cases: list[Case[str, set[str], None]] = []
    all_themes: set[str] = set()

    for row in df.to_dict("records"):
        response_id = row.get("ResponseID")
        if response_id is None:
            continue

        comment = normalize_text(row.get("Comment"))
        expected: set[str] = set()
        for column in ["theme_primary", "theme_secondary"]:
            value = row.get(column)
            if pd.isna(value):
                continue
            theme_name = normalize_text(value)
            if not theme_name or theme_name.lower() == "nan":
                continue
            if theme_name == "Other / unclear":
                continue
            expected.add(theme_name)

        all_themes.update(expected)
        cases.append(Case(name=str(int(response_id)), inputs=comment, expected_output=expected))

    return cases, sorted(all_themes)


def assert_input_matches_gold(
    gold_cases: list[Case[str, set[str], None]],
    input_csv_path: str | Path,
) -> None:
    """Assert each saved ThemeFinder input comment matches the gold comment."""
    saved_comments: set[str] = set()
    with open(input_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            saved_comments.add(extract_comment_from_response_text(row.get("response")))

    assert len(saved_comments) == len(gold_cases), (
        "The saved ThemeFinder input rows should match the reviewed gold rows one-for-one. "
        "We intentionally ignore the raw response_id because the input file and gold file "
        "are offset by one in some runs."
    )

    for case in gold_cases:
        assert case.inputs in saved_comments, (
            f"Comment for ResponseID {case.name} was not found in the saved ThemeFinder input. "
            f"This check intentionally matches on text, not on the raw response_id."
        )


def build_prediction_lookup(json_path: str | Path) -> dict[str, set[str]]:
    """Map each saved ThemeFinder response comment to the gold theme names it predicts."""
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    lookup: dict[str, set[str]] = {}
    for row in data.get("result", {}).get("mapping", []):
        response_text = row.get("response", "")
        comment = extract_comment_from_response_text(response_text)
        labels = row.get("labels") or []

        mapped = set()
        for label in labels:
            gold_theme = THEMEFINDER_ID_TO_GOLD.get(label)
            if gold_theme is not None:
                mapped.add(gold_theme)

        if comment:
            lookup[comment] = mapped

    return lookup


PREDICTIONS_BY_COMMENT = build_prediction_lookup(latest_themefinder_json("runs"))


def task(response_text: str) -> set[str]:
    """Return the gold-theme prediction set for one response using the saved run."""
    comment = extract_comment_from_response_text(response_text)
    return PREDICTIONS_BY_COMMENT.get(comment, set())


@dataclass
class ThemeF1(Evaluator[str, set[str], dict[str, Any] | None]):
    """F1 score for a set-of-themes prediction versus expected gold themes."""

    theme_names: list[str]

    def evaluate(
        self,
        ctx: EvaluatorContext[str, set[str], dict[str, Any] | None],
    ) -> float:
        expected = set(ctx.expected_output or set())
        predicted = set(ctx.output or set())

        # When both sides are empty, we treat the case as a perfect match so the
        # score stays defined instead of penalising neutral cases.
        if not expected and not predicted:
            return 1.0

        y_true = [1 if theme in expected else 0 for theme in self.theme_names]
        y_pred = [1 if theme in predicted else 0 for theme in self.theme_names]
        return f1(y_true, y_pred)


def per_theme_metrics(
    cases: list[Case[str, set[str], None]],
    predictions_by_case: dict[str, set[str]],
) -> list[tuple[str, int, int, float, float, float]]:
    """Return per-theme metrics sorted by gold count, low-count themes first."""
    gold_counts = Counter()
    for case in cases:
        for theme in case.expected_output:
            gold_counts[theme] += 1

    theme_names = sorted(gold_counts, key=lambda theme: (gold_counts[theme], theme))
    results: list[tuple[str, int, int, float, float, float]] = []

    for theme in theme_names:
        y_true = [1 if theme in case.expected_output else 0 for case in cases]
        y_pred = [1 if theme in predictions_by_case.get(case.name, set()) else 0 for case in cases]
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        score = f1(y_true, y_pred)
        pred_count = sum(1 for case in cases if theme in predictions_by_case.get(case.name, set()))
        results.append((theme, gold_counts[theme], pred_count, p, r, score))

    return results


def main() -> None:
    """Run the offline ThemeFinder evaluation and print the report."""
    gold_csv = Path("data/golden/initial_annotation_sample_reviewed.csv")
    input_csv = Path("data/golden/themefinder_input.csv")

    gold_cases, theme_names = load_gold_cases(gold_csv)
    assert_input_matches_gold(gold_cases, input_csv)

    dataset = Dataset(
        name="ThemeFinder offline mapping evaluation",
        cases=gold_cases,
        evaluators=[ThemeF1(theme_names=theme_names)],
    )
    report = dataset.evaluate_sync(task, progress=False, name="ThemeFinder theme mapping")
    print(report)

    predictions_by_case = {case.name: task(case.inputs) for case in gold_cases}
    print("\nPer-theme precision / recall / F1 (sorted by gold examples, smallest first):")
    print(f"{'theme':<45} {'gold':>5} {'pred':>5} {'precision':>10} {'recall':>10} {'f1':>10}")
    print("-" * 95)
    for theme, gold_count, pred_count, p, r, score in per_theme_metrics(gold_cases, predictions_by_case):
        print(
            f"{theme:<45} {gold_count:>5} {pred_count:>5} {p:>10.3f} {r:>10.3f} {score:>10.3f}"
        )


if __name__ == "__main__":
    main()
