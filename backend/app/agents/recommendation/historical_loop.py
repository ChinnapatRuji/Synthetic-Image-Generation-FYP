"""
Historical evaluation integration test.

Uses real saved YOLO evaluation results.
Does not train SDXL or YOLO.
Does not measure hyperparameter performance.
"""

from pathlib import Path
import optuna
import pandas as pd

from app.agents.recommendation.recommender import (
    suggest_hyperparameters,
    create_study,
)

from app.agents.recommendation.evaluator_adapter import (
    evaluation_to_score,
)

BACKEND = Path(__file__).resolve().parents[3]
RESULTS = BACKEND / "results" / "historical_evaluation"


def load_historical_summary(csv_path):
    df = pd.read_csv(csv_path)

    required = {"intended", "pass"}
    if not required.issubset(df.columns):
        raise ValueError("Missing evaluation columns")

    if df.empty:
        raise ValueError("Evaluation CSV is empty")

    passed = (
        df["pass"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
        })
    )

    if passed.isna().any():
        raise ValueError("Invalid pass/fail values")

    df["pass_numeric"] = passed

    classes = []

    for class_id, group in df.groupby("intended"):
        classes.append({
            "class_id": class_id,
            "pass_rate": float(group["pass_numeric"].mean()),
        })

    return {"classes": classes}


def objective(trial):
    params = suggest_hyperparameters(trial)

    # Historical datasets are selected independently
    # of the suggested hyperparameters.
    csv_path = RESULTS / (
        "txt2img_eval.csv"
        if trial.number % 2 == 0
        else "img2img_eval.csv"
    )

    summary = load_historical_summary(csv_path)
    score = evaluation_to_score(summary)

    trial.set_user_attr("evaluation_source", csv_path.name)
    trial.set_user_attr("mode", "HISTORICAL_INTEGRATION_ONLY")

    print(f"\nTrial {trial.number}")
    print("Recommended:", params)
    print("Historical dataset:", csv_path.name)
    print("Balanced pass rate:", round(score, 4))

    return score


if __name__ == "__main__":
    study = create_study(
        RESULTS / "historical_integration.db",
        study_name="historical_integration",
    )

    study.optimize(objective, n_trials=4)

    print("\nINTEGRATION TEST COMPLETE")
    print("Completed trials:", len(study.trials))
    print("Historical scores successfully recorded.")

    study.trials_dataframe().to_csv(
        RESULTS / "historical_trials.csv",
        index=False,
    )