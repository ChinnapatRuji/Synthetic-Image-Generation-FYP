import optuna
from pathlib import Path
import json

def suggest_hyperparameters(trial):
    """
    Recommend SDXL-LoRA training hyperparameters.
    """

    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-6,
            5e-4,
            log=True
        ),

        "network_dim": trial.suggest_categorical(
            "network_dim",
            [4, 8, 16, 32]
        ),

        "max_train_epochs": trial.suggest_int(
            "max_train_epochs",
            10,
            50
        )
    }


def create_study(database_path, study_name="sdxl_hpo"):
    """
    Create or resume an Optuna optimisation study.
    """

    database_path = Path(database_path).resolve()
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return optuna.create_study(
        direction="maximize",

        sampler=optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=3
        ),

        storage=f"sqlite:///{database_path.as_posix()}",

        study_name=study_name,

        load_if_exists=True
    )
    

def save_config(config, trial_number):
    output_dir = (
        Path(__file__).resolve().parent
        / "results"
        / f"trial_{trial_number:03d}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    #config_path = output_dir / "config.json"
    config_path = output_dir / "mock_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return config_path
    