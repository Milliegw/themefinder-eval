import json
from pathlib import Path
import pandas as pd

def load_runs(runs_dir: str | Path = "runs"):
    """Load all ThemeFinder run JSON files from the given directory."""
    run_dir = Path(runs_dir)
    run_files = sorted(run_dir.glob("run_*.json"))
    return [(path, json.loads(path.read_text())) for path in run_files]


def summarize_run(run, filename: str):
    """Return a compact summary for one run."""
    result = run.get("result", {})
    themes = result.get("themes") or []
    topics = [
        theme.get("topic", "")
        for theme in themes
        if isinstance(theme, dict) and theme.get("topic")
    ]
    mentions_fares = any("fare" in topic.lower() for topic in topics)

    return {
        "filename": filename,
        "timestamp": run.get("timestamp", "unknown"),
        "model": run.get("model", "unknown"),
        "temperature": run.get("temperature", "unknown"),
        "theme_count": len(themes),
        "topics": topics,
        "mentions_fares": mentions_fares,
    }


def main():
    runs = load_runs()
    if not runs:
        print("No run_*.json files found in runs/")
        return

    summaries = []
    for path, run in runs:
        summary = summarize_run(run, path.name)
        summaries.append(summary)
        print(f"{summary['filename']}: {summary['theme_count']} themes, "
              f"fares: {'yes' if summary['mentions_fares'] else 'no'}")

    table = pd.DataFrame(summaries)
    table["topics"] = table["topics"].apply("; ".join)   # list -> one readable cell
    table.to_csv("runs_summary.csv", index=False)
    print(f"Saved runs_summary.csv ({len(table)} runs)")


if __name__ == "__main__":
    main()
