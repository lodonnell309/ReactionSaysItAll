import optuna
import os
import yaml
import torch
import subprocess
from datetime import datetime
from pathlib import Path

# ==== Logging ====
log_dir = "outputs/logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "optuna_trials.csv")

# ==== Base config (mutable) ====
base_config = {
    "seed": 97,
    "data": {
        "root": "data/fer2013",
        "num_workers": 4,
    },
    "train": {
        "num_epochs": 50,
        "checkpoint_dir": "outputs/checkpoints/",
        "log_dir": log_dir,
    },
    "network": {
        "model": "cnn",
        "num_classes": 7,
        "input_size": 48,
    },
    "loss": {
        "use_class_weights": True,
        "class_counts": [3995, 436, 4097, 7215, 4965, 4830, 3171]
    }
}

# ==== Objective function for Optuna ====
def objective(trial):
    config = base_config.copy()

    # Suggest hyperparameters
    config["network"]["dropout"] = trial.suggest_float("dropout", 0.2, 0.6)
    config["train"]["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config["data"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    config["train"]["early_stop_patience"] = trial.suggest_int("early_stop_patience", 2, 5)

    # Save YAML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    cfg_path = f"configs/optuna/optuna_{timestamp}.yaml"
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)

    # Run subprocess
    try:
        subprocess.run(["python", "train.py", "--config_file", cfg_path], check=True)
        # Assume training script logs best val_acc in CSV last row
        import pandas as pd
        df = pd.read_csv(os.path.join(config["train"]["log_dir"], "experiment_log.csv"))
        val_acc = df.iloc[-1]["val acc"]
        return val_acc
    except subprocess.CalledProcessError:
        return 0.0  # fallback on crash

# ==== Run Study ====
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\nBest Trial:")
    print(study.best_trial)

    # Optionally save study results
    with open(os.path.join(log_dir, "optuna_best.yaml"), "w") as f:
        yaml.dump(study.best_trial.params, f)
