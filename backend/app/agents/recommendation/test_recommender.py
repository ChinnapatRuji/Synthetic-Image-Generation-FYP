from pathlib import Path
import math
import json

from recommender import (
    suggest_hyperparameters,
    create_study,
    save_config
)


# def mock_evaluator(config):
#     """
#     Artificial score for testing the optimization loop.
#     """
#     lr = config["learning_rate"]
#     rank = config["network_dim"]

#     lr_penalty = abs(
#         math.log10(lr) - math.log10(5e-5)
#     )

#     rank_penalty = abs(rank - 16) / 16

#     return 100 - 20 * lr_penalty - 10 * rank_penalty
def mock_evaluator(config):
    lr = config["learning_rate"]
    rank = config["network_dim"]
    epochs = config["max_train_epochs"]

    lr_penalty = abs(
        math.log10(lr / 5e-5)
    )

    rank_penalty = abs(rank - 16) / 16

    epoch_penalty = abs(epochs - 30) / 30

    return (
        100
        - 20 * lr_penalty
        - 10 * rank_penalty
        - 10 * epoch_penalty
    )

def objective(trial):

    config = suggest_hyperparameters(trial)

    config_path = save_config(
        config,
        trial.number
    )

    score = mock_evaluator(config)

    print(f"\nTrial {trial.number}")
    print("Configuration:", config)
    print("Saved to:", config_path)
    print("Score:", round(score, 2))

    return score



DATABASE_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "optuna_study.db"
)

study = create_study(database_path=DATABASE_PATH)

study.optimize(objective, n_trials=10)

print("\nOptimization complete!")
print("Best configuration:", study.best_params)
print("Best score:", study.best_value)
