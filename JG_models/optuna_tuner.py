import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.trainer_cnn import CNNTrainer
from utils.config_loader import Config
from utils.seed import set_seed
import yaml
import os

def objective(trial):
    config_dict = {
        "seed": trial.suggest_categorical("seed", [42, 97, 112358]),
        "data": {
            "root": "data/fer2013",
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "num_workers": 4
        },
        "train": {
            "num_epochs": 30,
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "early_stop_patience": 5,
            "checkpoint_dir": "outputs/checkpoints/",
            "log_dir": "outputs/logs/"
        },
        "network": {
        "model": "tunable",
        "model_file": "optuna_tunable",      # ← NEW: tells trainer where to import
        "model_class": "TunableCNN",         # ← NEW: tells trainer which class to use
        "num_classes": 7,
        "input_size": 48,
        "dropout": trial.suggest_float("dropout", 0.2, 0.5)
    },
        "loss": {
            "use_class_weights": True,
            "class_counts": [750, 436, 250, 1995, 4230, 4243, 3171]
        }
    }

    config = Config(config_dict)
    set_seed(config.seed)

    trainer = CNNTrainer(config)

    trainer.train()

    return trainer.best_val_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    print("Best Trial:")
    print(study.best_trial)
    os.makedirs("optuna_logs", exist_ok=True)
    with open("optuna_logs/best_config.yaml", "w") as f:
        yaml.dump(study.best_trial.params, f)
