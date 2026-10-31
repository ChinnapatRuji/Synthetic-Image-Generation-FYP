"""
CPU-only integration prototype.

Connects:
Recommender -> Evaluation Agent -> Evaluator Adapter -> Optuna

No SDXL training or generation is performed.
No YOLO training is performed.
"""

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


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = BACKEND_DIR.parent

RESULTS_DIR = BACKEND_DIR / "results" / "recommender"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CHANGE THESE TWO PATHS TO YOUR ACTUAL FILES
YOLO_WEIGHTS = PROJECT_ROOT / "best.pt"

GENERATED_DIR = PROJECT_ROOT / "generated" / "txt2img"


# --------------------------------------------------
# 2. LOAD GENERATED IMAGES
# --------------------------------------------------

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


# --------------------------------------------------
# 3. OPTUNA OBJECTIVE
# --------------------------------------------------

def objective(trial):

    print(f"\n========== TRIAL {trial.number} ==========")

    # STEP A: Recommend hyperparameters
    params = suggest_hyperparameters(trial)

    print("Recommended hyperparameters:", params)

    # STEP B: Save trial configuration
    trial_dir = RESULTS_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    config_path = trial_dir / "recommended_params.json"

    config_path.write_text(
        json.dumps(params, indent=2),
        encoding="utf-8",
    )

    # STEP C: Load existing generated images
    images, class_names = load_generated_images(
        GENERATED_DIR
    )

    print("Images loaded:", len(images))

    # STEP D: Run actual YOLO Evaluation Agent
    rows = classify_images(
        weights=YOLO_WEIGHTS,
        images=images,
        imgsz=224,
    )

    # STEP E: Summarise evaluation results
    summary = summarize(
        rows=rows,
        class_names=class_names,
    )

    # STEP F: Convert evaluation to Optuna score
    score = evaluation_to_score(summary)

    # STEP G: Save evaluation results
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

    # STEP H: Return feedback to Optuna
    return score


# --------------------------------------------------
# 4. RUN OPTIMISATION LOOP
# --------------------------------------------------

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