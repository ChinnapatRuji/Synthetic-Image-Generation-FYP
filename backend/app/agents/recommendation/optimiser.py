from pathlib import Path
import json
import optuna

from app.agents.recommendation.recommender import (
    suggest_hyperparameters,
    create_study,
)

from app.agents.recommendation.evaluator_adapter import (
    evaluation_to_score,
)

from app.agents.evaluation.evaluation import (
    classify_images,
    summarize,
)

BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_DIR.parent

RESULTS_DIR = BACKEND_DIR / "results" / "recommender"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHTS = PROJECT_ROOT / "best.pt"

GENERATED_DIR = PROJECT_ROOT / "generated" / "txt2img"


def load_generated_images(directory):

    images = []
    class_names = {}

    for class_dir in sorted(directory.iterdir()):

        if not class_dir.is_dir():
            continue

        class_id = class_dir.name
        class_names[class_id] = class_id

        for image in sorted(class_dir.iterdir()):

            if image.suffix.lower() not in {
                ".png", ".jpg", ".jpeg", ".bmp", ".webp"
            }:
                continue

            images.append({
                "file": image.name,
                "path": image,
                "class_id": class_id,
            })

    if not images:
        raise ValueError("No generated images found")

    return images, class_names

def objective(trial):

    print(f"\n========== TRIAL {trial.number} ==========")

    params = suggest_hyperparameters(trial)

    print("Recommended hyperparameters:", params)

    trial_dir = RESULTS_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    config_path = trial_dir / "recommended_params.json"

    config_path.write_text(
        json.dumps(params, indent=2),
        encoding="utf-8",
    )

    images, class_names = load_generated_images(
        GENERATED_DIR
    )

    print("Images loaded:", len(images))

    rows = classify_images(
        weights=YOLO_WEIGHTS,
        images=images,
        imgsz=224,
    )

    summary = summarize(
        rows=rows,
        class_names=class_names,
    )

    score = evaluation_to_score(summary)

    evaluation_path = trial_dir / "evaluation.json"

    evaluation_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    trial.set_user_attr(
        "evaluation_mode",
        "FIXED_IMAGE_INTEGRATION_TEST",
    )

    trial.set_user_attr(
        "config_path",
        str(config_path),
    )

    print("Evaluation score:", score)

    return score


if __name__ == "__main__":

    if not YOLO_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {YOLO_WEIGHTS}"
        )

    if not GENERATED_DIR.is_dir():
        raise FileNotFoundError(
            f"Generated images not found: {GENERATED_DIR}"
        )

    study = create_study(
        RESULTS_DIR / "cpu_integration.db",
        study_name="cpu_integration",
    )

    study.optimize(
        objective,
        n_trials=3,
    )

    print("\n========== INTEGRATION COMPLETE ==========")

    print("Completed trials:",
          sum(
              t.state == optuna.trial.TrialState.COMPLETE
              for t in study.trials
          ))

    print("Best integration score:", study.best_value)

    print("Best trial parameters:", study.best_params)

    study.trials_dataframe().to_csv(
        RESULTS_DIR / "trial_history.csv",
        index=False,
    )