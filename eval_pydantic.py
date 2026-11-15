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

from compare_themefinder_gold import latest_themefinder_json, map_themefinder_label_to_codebook
from src.metrics import f1, precision, recall


def normalize_text(value: str | None) -> str:
    """Collapse whitespace so comment matching is stable across CSV and JSON inputs."""
    return " ".join(str(value or "").strip().split())


def load_gold_cases(csv_path: str | Path) -> tuple[list[Case[str, set[str], None]], list[str]]:
    """Build dataset cases from the reviewed gold CSV."""
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
    """Assert each saved ThemeFinder input row matches the reviewed gold IDs."""
    saved_response_ids: set[int] = set()
    with open(input_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            response_id = row.get("response_id")
            if response_id is not None:
                saved_response_ids.add(int(response_id))

    gold_ids = {int(case.name) for case in gold_cases}
    assert saved_response_ids == gold_ids, (
        "The saved ThemeFinder input rows should match the reviewed gold rows by response_id."
    )


def build_prediction_lookup(json_path: str | Path) -> dict[int, set[str]]:
    """Map each saved ThemeFinder response_id to the human gold theme names it predicts."""
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    theme_descriptions = {
        row.get("topic_id"): normalize_text(row.get("topic", ""))
        for row in data.get("result", {}).get("themes", [])
        if row.get("topic_id") is not None
    }

    lookup: dict[int, set[str]] = {}
    for row in data.get("result", {}).get("mapping", []):
        response_id = row.get("response_id")
        labels = row.get("labels") or []
        if response_id is None:
            continue

        mapped: set[str] = set()
        for label in labels:
            mapped.update(map_themefinder_label_to_codebook(label, theme_descriptions))

        lookup[int(response_id)] = mapped

    return lookup


def task(response_id: int) -> set[str]:
    """Return the gold-theme prediction set for one response using the saved run."""
    predictions_by_id = build_prediction_lookup(latest_themefinder_json("runs"))
    return predictions_by_id.get(int(response_id), set())


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

    predictions_by_case = {case.name: task(int(case.name)) for case in gold_cases}

    dataset = Dataset(
        name="ThemeFinder offline mapping evaluation",
        cases=gold_cases,
        evaluators=[ThemeF1(theme_names=theme_names)],
    )
    report = dataset.evaluate_sync(
        lambda case: predictions_by_case.get(str(case.name), set()),
        progress=False,
        name="ThemeFinder theme mapping",
    )
    print(report)
    print("\nPer-theme precision / recall / F1 (sorted by gold examples, smallest first):")
    print(f"{'theme':<45} {'gold':>5} {'pred':>5} {'precision':>10} {'recall':>10} {'f1':>10}")
    print("-" * 95)
    for theme, gold_count, pred_count, p, r, score in per_theme_metrics(gold_cases, predictions_by_case):
        print(
            f"{theme:<45} {gold_count:>5} {pred_count:>5} {p:>10.3f} {r:>10.3f} {score:>10.3f}"
        )


if __name__ == "__main__":
    main()
