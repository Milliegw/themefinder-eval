import argparse
import json
from pathlib import Path
from urllib import request, error


def call_ollama(model: str, prompt: str, url: str = "http://localhost:11434/api/generate") -> dict:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama at http://localhost:11434. "
            "Make sure the local Ollama service is running."
        ) from exc

    raw = data.get("response", "")
    if raw:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    return {"label": "flag_for_review", "reason": raw or "Could not parse model response."}


def build_prompt(comment: str) -> str:
    return (
        "You are reviewing consultation responses for a theme-annotation golden dataset.\n"
        "Decide whether this comment is useful for theme analysis.\n"
        "Return ONLY valid JSON with exactly these keys: label and reason.\n"
        "Use label = 'keep' if the comment contains a real opinion, problem, suggestion, or lived experience.\n"
        "Use label = 'flag_for_review' if the comment is vague, non-useful, or clearly not informative.\n"
        "Examples of flag_for_review: 'no idea', 'not sure', 'don't know', 'nothing to add', 'n/a'.\n"
        "Examples of keep: concrete problems, suggestions, reasons, experiences, policy concerns.\n"
        f"Comment: {comment}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run a local Ollama model to flag low-quality consultation comments.")
    parser.add_argument("--input", default="data/golden/initial_annotation_sample.csv", help="CSV file to review")
    parser.add_argument("--output", default="data/golden/initial_annotation_sample_llm.csv", help="Where to save the reviewed CSV")
    parser.add_argument("--model", default="llama3.1", help="Local Ollama model name")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional limit for testing on a small subset")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = input_path.read_text(encoding="utf-8-sig")
    # Read the CSV with pandas so we retain the original structure.
    import pandas as pd

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    results = []
    for _, row in df.iterrows():
        comment = str(row.get("Comment", "")).strip()
        if not comment:
            result = {"label": "flag_for_review", "reason": "Empty comment"}
        else:
            prompt = build_prompt(comment)
            result = call_ollama(args.model, prompt)

        result_row = {
            "Outcome": row.get("Outcome", ""),
            "Question": row.get("Question", ""),
            "Comment": comment,
            "llm_label": result.get("label", "flag_for_review"),
            "llm_reason": result.get("reason", ""),
        }
        results.append(result_row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)

    print(f"Saved {len(out_df)} reviewed rows to {output_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()
