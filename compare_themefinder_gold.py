import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.metrics import bootstrap_ci, f1, precision, recall

# Use a stable codebook for the final taxonomy rather than the run-specific topic IDs
# emitted by ThemeFinder. The model's A–H labels can change between runs, so our
# evaluation should anchor on the human theme names we actually care about.
THEME_CODEBOOK = {
    "Children and young people": {"children", "young", "youth", "school", "academy", "nursery"},
    "Education and skills": {"education", "school", "skills", "teacher", "academies"},
    "Community and local services": {"community", "local services", "support", "signposting", "families"},
    "Community safety and crime": {"safety", "crime", "policing", "anti-social", "violence"},
    "Public services and council delivery": {"council", "public services", "service delivery", "delivery", "department"},
    "Housing and homelessness": {"housing", "homeless", "affordable housing", "accommodation", "rent"},
    "Jobs and local economy": {"jobs", "economy", "employment", "business", "local economy"},
    "Transport and travel": {"transport", "travel", "roads", "cycling", "car", "accessibility"},
    "Environment and climate": {"environment", "climate", "waste", "recycling", "green", "zero waste"},
    "Health and wellbeing": {"health", "wellbeing", "mental health", "food", "hospital", "care"},
    "Equality, inclusion, and access": {"equality", "inclusion", "access", "disabled", "accessible"},
}

# Compatibility aliases for the earlier A-H mapping approach. We intentionally keep
# the theme names stable and human-readable rather than depending on the run-specific
# labels emitted by ThemeFinder itself.
GOLD_TO_THEMEFINDER_ID = {theme_name: theme_name for theme_name in THEME_CODEBOOK}
THEMEFINDER_TOPIC_LABELS = {theme_name: theme_name for theme_name in THEME_CODEBOOK}


def normalize_theme_name(value: object) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def map_themefinder_label_to_codebook(label: object, theme_descriptions: dict[str, str]) -> set[str]:
    raw_label = normalize_theme_name(label)
    if not raw_label:
        return set()

    label_key = raw_label.lower()
    description = " ".join(
        normalize_theme_name(theme_descriptions.get(raw_label, "")).split()
    )
    haystack = f"{raw_label} {description}".lower()

    scored_matches: list[tuple[int, str]] = []
    for theme_name, keywords in THEME_CODEBOOK.items():
        score = 0
        if theme_name.lower() == label_key:
            score += 100
        if theme_name.lower() in haystack:
            score += 30
        for keyword in keywords:
            kw = keyword.lower()
            if kw in haystack:
                score += len(kw) + (4 if len(kw.split()) > 1 else 0)
        if score > 0:
            scored_matches.append((score, theme_name))

    if scored_matches:
        top_score = max(score for score, _ in scored_matches)
        winners = {theme_name for score, theme_name in scored_matches if score == top_score}
        return winners

    # Falls back to a direct lookup for a human theme name that is already in the codebook.
    if raw_label in THEME_CODEBOOK:
        return {raw_label}

    return set()


def latest_themefinder_json(runs_dir: str | Path = "runs") -> Path:
    run_dir = Path(runs_dir)
    files = sorted(run_dir.glob("themefinder_test_*.json"))
    if not files:
        raise FileNotFoundError(f"No themefinder_test_*.json files found in {run_dir}")
    return files[-1]


def load_model_predictions(json_path: str | Path) -> dict[int, set[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    theme_descriptions = {
        row.get("topic_id"): normalize_theme_name(row.get("topic", ""))
        for row in data.get("result", {}).get("themes", [])
        if row.get("topic_id") is not None
    }

    predictions: dict[int, set[str]] = {}
    for row in data.get("result", {}).get("mapping", []):
        response_id = row.get("response_id")
        if response_id is None:
            continue

        labels = row.get("labels", []) or []
        mapped: set[str] = set()
        for label in labels:
            mapped.update(map_themefinder_label_to_codebook(label, theme_descriptions))
        predictions[int(response_id)] = mapped

    return predictions


def load_gold_labels(csv_path: str | Path) -> dict[int, set[str]]:
    df = pd.read_csv(csv_path)
    gold: dict[int, set[str]] = {}

    for row in df.to_dict("records"):
        response_id = row.get("ResponseID")
        if response_id is None:
            continue

        themes = set()
        for column in ["theme_primary", "theme_secondary"]:
            value = row.get(column)
            if pd.isna(value):
                continue
            cleaned = normalize_theme_name(value)
            if cleaned and cleaned.lower() != "nan" and cleaned != "Other / unclear":
                themes.add(cleaned)

        gold[int(response_id)] = themes

    return gold


def format_theme_ids(themes: set[str]) -> str:
    if not themes:
        return "—"
    return ", ".join(sorted(themes))


def display_theme_name(theme_id: str) -> str:
    label = THEMEFINDER_TOPIC_LABELS.get(theme_id, theme_id)
    if label == theme_id:
        return str(theme_id)
    return f"{theme_id} ({label})"


def print_side_by_side(predictions: dict[int, set[str]], gold: dict[int, set[str]], max_rows: int = 20) -> None:
    all_ids = sorted(set(predictions) | set(gold))
    print(f"\nSide-by-side comparison (showing first {min(max_rows, len(all_ids))} responses):")
    print(f"{'ResponseID':>10} | {'Gold theme IDs':<35} | {'Predicted theme IDs':<35} | {'Status'}")
    print("-" * 110)

    for response_id in all_ids[:max_rows]:
        gold_themes = gold.get(response_id, set())
        pred_themes = predictions.get(response_id, set())
        status = "MATCH" if gold_themes == pred_themes else "DIFFERENT"
        print(
            f"{response_id:>10} | {format_theme_ids(gold_themes):<35} | "
            f"{format_theme_ids(pred_themes):<35} | {status}"
        )


def build_markdown_report(
    predictions: dict[int, set[str]],
    gold: dict[int, set[str]],
    json_path: str | Path,
    csv_path: str | Path,
    max_rows: int = 20,
) -> str:
    all_ids = sorted(set(predictions) | set(gold))
    all_themes = sorted(
        set().union(*(set(v) for v in predictions.values() if v))
        | set().union(*(set(v) for v in gold.values() if v))
    )

    lines = [
        "# ThemeFinder vs Gold Comparison (ThemeFinder taxonomy)",
        "",
        f"- ThemeFinder JSON: `{json_path}`",
        f"- Gold CSV: `{csv_path}`",
        f"- Responses compared: {len(all_ids)}",
        f"- Themes compared: {len(all_themes)}",
        "",
        "## Per-theme metrics",
        "",
        "| Theme | Label | Precision | Recall | F1 | F1 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for theme in all_themes:
        y_true = [1 if theme in gold.get(rid, set()) else 0 for rid in all_ids]
        y_pred = [1 if theme in predictions.get(rid, set()) else 0 for rid in all_ids]
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        f = f1(y_true, y_pred)
        low, high = bootstrap_ci(y_true, y_pred, f1)
        lines.append(
            f"| {theme} | {display_theme_name(theme)} | {p:.3f} | {r:.3f} | {f:.3f} | ({low:.3f}, {high:.3f}) |"
        )

    lines.extend([
        "",
        "## Overall label counts",
        "",
        "| Theme | Label | Gold count | Predicted count |",
        "| --- | --- | ---: | ---: |",
    ])

    for theme in all_themes:
        pred_count = sum(1 for rid in all_ids if theme in predictions.get(rid, set()))
        gold_count = sum(1 for rid in all_ids if theme in gold.get(rid, set()))
        lines.append(f"| {theme} | {display_theme_name(theme)} | {gold_count} | {pred_count} |")

    lines.extend([
        "",
        "## Side-by-side examples",
        "",
        "| ResponseID | Gold theme IDs | Predicted theme IDs | Status |",
        "| ---: | --- | --- | --- |",
    ])

    for response_id in all_ids[:max_rows]:
        gold_themes = gold.get(response_id, set())
        pred_themes = predictions.get(response_id, set())
        status = "MATCH" if gold_themes == pred_themes else "DIFFERENT"
        lines.append(
            f"| {response_id} | {format_theme_ids(gold_themes)} | {format_theme_ids(pred_themes)} | {status} |"
        )

    return "\n".join(lines) + "\n"


def write_report(
    predictions: dict[int, set[str]],
    gold: dict[int, set[str]],
    json_path: str | Path,
    csv_path: str | Path,
    max_rows: int = 20,
) -> Path:
    run_dir = Path("runs")
    run_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = run_dir / f"themefinder_gold_comparison_{timestamp}.md"
    report_path.write_text(
        build_markdown_report(predictions, gold, json_path, csv_path, max_rows=max_rows),
        encoding="utf-8",
    )
    return report_path


def print_metrics(predictions: dict[int, set[str]], gold: dict[int, set[str]]) -> None:
    all_ids = sorted(set(predictions) | set(gold))
    all_themes = sorted(
        set().union(*(set(v) for v in predictions.values() if v))
        | set().union(*(set(v) for v in gold.values() if v))
    )

    print(f"Responses compared: {len(all_ids)}")
    print(f"Themes compared: {len(all_themes)}")
    print()

    for theme in all_themes:
        y_true = [1 if theme in gold.get(rid, set()) else 0 for rid in all_ids]
        y_pred = [1 if theme in predictions.get(rid, set()) else 0 for rid in all_ids]

        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        f = f1(y_true, y_pred)
        low, high = bootstrap_ci(y_true, y_pred, f1)

        print(
            f"{theme:<8} {display_theme_name(theme):<45} precision={p:.3f} "
            f"recall={r:.3f} f1={f:.3f} f1_95ci=({low:.3f}, {high:.3f})"
        )

    # Simple overall counts for context
    print()
    print("Overall label counts:")
    for theme in all_themes:
        pred_count = sum(1 for rid in all_ids if theme in predictions.get(rid, set()))
        gold_count = sum(1 for rid in all_ids if theme in gold.get(rid, set()))
        print(f"  {theme}: {display_theme_name(theme)} | gold={gold_count}, predicted={pred_count}")


def main():
    parser = argparse.ArgumentParser(description="Compare ThemeFinder predictions to the human gold labels.")
    parser.add_argument(
        "--json",
        default=None,
        help="Path to the ThemeFinder JSON output. Defaults to the newest themefinder_test_*.json in runs/.",
    )
    parser.add_argument(
        "--csv",
        default="data/golden/initial_annotation_sample_reviewed.csv",
        help="Path to the reviewed gold CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20,
        help="Number of response rows to print in the side-by-side comparison table.",
    )
    args = parser.parse_args()

    if args.json:
        json_path = Path(args.json)
    else:
        json_path = latest_themefinder_json("runs")

    predictions = load_model_predictions(json_path)
    gold = load_gold_labels(args.csv)

    print(f"Using ThemeFinder file: {json_path}")
    print(f"Using gold file: {args.csv}")
    print(f"Using taxonomy: ThemeFinder topic IDs")
    print()
    print_metrics(predictions, gold)
    print_side_by_side(predictions, gold, max_rows=args.max_rows)

    report_path = write_report(predictions, gold, json_path, args.csv, max_rows=args.max_rows)
    print(f"\nComparison report written to: {report_path}")


if __name__ == "__main__":
    main()
