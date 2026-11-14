import optuna
from pathlib import Path
import json
from pydantic import BaseModel, Field, ConfigDict, model_validator

# Recommend which hyperparameter to try
def suggest_hyperparameters(trial):

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

#check recommended hyperparameters
def validate_config(config):
    
    validated = RecommendedConfig.model_validate(config)
    return validated.model_dump()

# save config as JSON file
def save_config(config, trial_number):
    output_dir = (
        Path(__file__).resolve().parent
        / "results"
        / f"trial_{trial_number:03d}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "mock_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return config_path

# optuna record trial results
def create_study(database_path, study_name="sdxl_hpo"):

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
    
# define validation schema for recommended hyperparameters
class RecommendedConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(
        ge=1e-6,
        le=5e-4
    )

    network_dim: int = Field(
        ge=4,
        le=32
    )

    max_train_epochs: int = Field(
        ge=10,
        le=50
    )

    @model_validator(mode="after")
    def check_rank(self):
        if self.network_dim not in [4, 8, 16, 32]:
            raise ValueError("Invalid LoRA rank")
        return self
