from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Step 1: Clean the raw consultation data.
# Remove rows that do not contain usable text.
# ------------------------------------------------------------
def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with valid comment text."""
    cleaned = df.copy()

    # Drop rows where the comment is missing.
    cleaned = cleaned.dropna(subset=["Comment"])

    # Trim whitespace around each comment.
    cleaned["Comment"] = cleaned["Comment"].astype(str).str.strip()

    # Remove empty comments.
    cleaned = cleaned[cleaned["Comment"] != ""]

    # Normalise repeated whitespace/newlines so the text is consistent.
    cleaned["Comment"] = cleaned["Comment"].str.replace(r"\s+", " ", regex=True)

    return cleaned


# ------------------------------------------------------------
# Step 2: Rule-based low-quality flagging.
# Before sampling, flag obvious non-useful comments such as
# "no idea", "not sure" or "nothing to add".
# This is a first pass, not the final decision.
# ------------------------------------------------------------
def flag_low_quality_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Add a usefulness flag for obvious low-quality responses."""
    flagged = df.copy()

    # Lowercase version for text matching.
    flagged["comment_clean"] = flagged["Comment"].fillna("").astype(str).str.lower().str.strip()

    low_quality_phrases = [
        "don't know",
        "dont know",
        "no idea",
        "not sure",
        "not applicable",
        "n/a",
        "no comment",
        "nothing to add",
        "no opinion",
        "can't think of anything",
        "not relevant",
    ]

    # Default to keep, then flag obvious low-quality responses.
    flagged["usefulness_flag"] = "keep"
    low_quality_mask = flagged["comment_clean"].str.contains(
        "|".join(low_quality_phrases),
        regex=True,
    )
    flagged.loc[low_quality_mask, "usefulness_flag"] = "flag_for_review"

    return flagged


# ------------------------------------------------------------
# Step 3: Build a balanced sample for annotation.
# We take a representative set across Outcome groups.
# ------------------------------------------------------------
def build_initial_sample(df: pd.DataFrame, n_per_group: int = 50) -> pd.DataFrame:
    """Take a balanced sample across Outcome groups for annotation."""
    sample = (
        df.groupby("Outcome", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_group, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    return sample


# ------------------------------------------------------------
# Step 4: Main script entry point.
# This loads the CSV, flags weak responses, creates a sample,
# and saves it as a manually reviewable annotation set.
# ------------------------------------------------------------
def main():
    # Set to True to remove low-quality responses automatically.
    # Set to False to keep them for manual review before deciding.
    drop_low_quality = False

    # Path to the raw consultation data.
    input_path = Path("data") / "council-plan-consultation-19-report-annex.csv"

    # Folder to save the annotation-ready sample.
    output_dir = Path("data") / "golden"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "initial_annotation_sample.csv"

    # Read the raw CSV file.
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Record original dataset size before filtering.
    original_rows = len(df)

    # Step 1: clean obvious junk.
    df_clean = clean_comments(df)

    # Step 2: flag low-quality comments using simple rules.
    df_flagged = flag_low_quality_comments(df_clean)

    # Step 3: decide whether to drop flagged rows before sampling.
    if drop_low_quality:
        df_flagged = df_flagged[df_flagged["usefulness_flag"] == "keep"]

    # Step 4: build a balanced sample for human annotation.
    sample = build_initial_sample(df_flagged, n_per_group=50)

    # Save the sample for annotation.
    sample.to_csv(output_path, index=False)

    # Print a quick summary for the terminal.
    print(f"Original rows: {original_rows}")
    print(f"Cleaned rows: {len(df_clean)}")
    print(f"Rows flagged for review: {len(df_flagged[df_flagged['usefulness_flag'] == 'flag_for_review'])}")
    print(f"Sample rows: {len(sample)}")
    print(f"Saved sample to: {output_path}")
    print(sample.head())


if __name__ == "__main__":
    main()

