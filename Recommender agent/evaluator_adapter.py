
from pathlib import Path
import pandas as pd


def read_evaluation_score(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Evaluation CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Evaluation CSV is empty")

    if "pass" not in df.columns:
        raise ValueError(
            "Evaluation CSV must contain a 'pass' column"
        )

    passed = (
        df["pass"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0
        })
    )

    if passed.isna().any():
        raise ValueError(
            "Invalid values in the 'pass' column"
        )

    return float(passed.mean())
