import json
import sys
from pathlib import Path
import optuna
import os
import subprocess
from recommender import (
    suggest_hyperparameters,
    create_study,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Recommender agent" / "results"
BASE_CONFIG = ROOT / "config.json"

sys.path.insert(0, str(ROOT))
from config import load_config


def build_trial_config(params, trial_number):
    # Read existing project configuration
    config = json.loads(
        BASE_CONFIG.read_text(encoding="utf-8")
    )

    trial_name = f"trial_{trial_number:03d}"
    trial_dir = RESULTS / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Map Optuna parameters into existing config schema
    config["training"]["optimizer"]["unet_lr"] = (
        params["learning_rate"]
    )

    config["lora"]["network_dim"] = params["network_dim"]
    config["lora"]["network_alpha"] = params["network_dim"]

    config["training"]["epochs"] = params["max_train_epochs"]
    config["training"]["max_train_steps"] = 0

    # Unique training outputs
    config["training"]["output_name"] = trial_name
    config["generate"]["output_name"] = trial_name

    config["paths"]["output_dir"] = str(
        trial_dir / "checkpoints"
    )

    config["paths"]["generated_dir"] = str(
        trial_dir / "generated"
    )

    # Save full trial configuration
    config_path = trial_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    # Validate using your project's Pydantic schema
    load_config(config_path)

    return config_path, trial_dir

def preview_training(config_path):
    env = os.environ.copy()

    env["FYP_CONFIG"] = str(config_path.resolve())
    env["FYP_PREVIEW"] = "1"

    subprocess.run(
        [sys.executable, str(ROOT / "train_sd_lora_kohya.py")],
        cwd=ROOT,
        env=env,
        check=True
    )

def objective(trial):
    # 1. Recommend hyperparameters
    params = suggest_hyperparameters(trial)

    # 2. Create a complete training configuration
    config_path, trial_dir = build_trial_config(
        params, trial.number
    )

    # 3. Record trial information
    trial.set_user_attr("config_path", str(config_path))

    print(f"\n========== TRIAL {trial.number} ==========")
    print("Recommended:", params)
    print("Configuration:", config_path)

    # 4. Validate configuration
    cfg = load_config(config_path)

    print("Configuration valid!")
    print("\nPreparing Kohya training command...")
    
    preview_training(config_path)
    
    trial.set_user_attr("config_valid", True)
    trial.set_user_attr("training_preview", True)
    trial.set_user_attr("mode", "configuration_test")

    print("Optimizer:", cfg.training.optimizer.name)
    print("Learning rate:", cfg.training.optimizer.unet_lr)
    print("LoRA rank:", cfg.lora.network_dim)
    print("Epochs:", cfg.training.epochs)

    # 5. No training is performed in this mode.
    # We deliberately do not return a fake quality score.
    #trial.set_user_attr("mode", "configuration_test")

    # Tell Optuna this trial was not evaluated.
    raise optuna.TrialPruned(
        "Configuration validated; training not executed."
    )


if __name__ == "__main__":
    

    study = create_study(
        RESULTS / "configuration_test.db",
        study_name="configuration_test"
    )

    study.optimize(objective, n_trials=3)

    print("\nConfiguration test completed.")
    print("Trials recorded:", len(study.trials))

    for trial in study.trials:
        print(
            f"Trial {trial.number}: "
            f"{trial.state.name} | "
            f"{trial.params}"
        )


# def objective(trial):
#     params = suggest_hyperparameters(trial)

#     config_path, trial_dir = build_trial_config(
#         params,
#         trial.number
#     )

#     print(f"\nTrial {trial.number}")
#     print("Recommended parameters:", params)
#     print("Configuration saved:", config_path)
#     print("Configuration validation: PASSED")

#     trial.set_user_attr(
#         "config_path",
#         str(config_path)
#     )

#     # Training and evaluation will be added later.
#     # No quality score is returned yet.
#     raise NotImplementedError(
#         "Configuration validated. "
#         "Training integration is not enabled."
#     )


# if __name__ == "__main__":
#     study = create_study(
#         RESULTS / "sd_hpo.db",
#         study_name="sd_hpo"
#     )

#     study.optimize(objective, n_trials=1)
