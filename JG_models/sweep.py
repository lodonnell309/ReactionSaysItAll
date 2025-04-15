import os
import itertools
import yaml
import subprocess
import pandas as pd
from datetime import datetime
import time

# ==== Sweep Config Space ====
search_space = {
    "dropout": [.27],
    "lr": [0.0086],
    "batch_size": [32, 64],
    "early_stop_patience": [5],
    "num_epochs": [50],
    "seed": [112358],
    "use_class_weights": [True, False],
    "loss_type": ["cross_entropy", "focal"],
    "focal_gamma": [0.001, 0.5, 1.0, 2.0],
}

# ==== Base Config ====
base_config = {
    "data": {
        "root": "data/fer2013",
        "num_workers": 4
    },
    "train": {
        "checkpoint_dir": "outputs/checkpoints/",
        "log_dir": "outputs/logs/"
    },
    "network": {
        "model": "cnn",
        "num_classes": 7,
        "input_size": 48
    },
    "loss": {
        "class_counts": [3995, 436, 4097, 7215, 4965, 4830, 3171]
    }
}

# ==== Output Setup ====
os.makedirs("configs/sweeps", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)
log_csv_path = "outputs/logs/experiment_log.csv"

if not os.path.exists(log_csv_path):
    df = pd.DataFrame(columns=["timestamp", "train_loss", "train_acc", "val_loss", "val_acc", "config"])
    df.to_csv(log_csv_path, index=False)

# ==== Sweep ====
keys, values = zip(*search_space.items())
combinations = list(itertools.product(*values))

for i, combo in enumerate(combinations):
    combo_dict = dict(zip(keys, combo))

    # Skip invalid combos (e.g., focal_gamma specified for non-focal loss)
    if combo_dict["loss_type"] != "focal" and combo_dict["focal_gamma"] != 0.0:
        continue

    print(f"\n🚀 Running combo {i+1}/{len(combinations)}: {combo_dict}")

    config = base_config.copy()
    config["data"]["batch_size"] = combo_dict["batch_size"]
    config["train"]["lr"] = combo_dict["lr"]
    config["train"]["early_stop_patience"] = combo_dict["early_stop_patience"]
    config["train"]["num_epochs"] = combo_dict["num_epochs"]
    config["network"]["dropout"] = combo_dict["dropout"]
    config["seed"] = combo_dict["seed"]
    config["loss"]["use_class_weights"] = combo_dict["use_class_weights"]
    config["loss"]["loss_type"] = combo_dict["loss_type"]
    if combo_dict["loss_type"] == "focal":
        config["loss"]["focal_gamma"] = combo_dict["focal_gamma"]

    # Save to YAML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    yaml_path = f"configs/sweeps/sweep_{timestamp}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    try:
        subprocess.run(["python", "train.py", "--config_file", yaml_path], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Training failed for {yaml_path} - skipping.")
        continue

    time.sleep(1)

print("\n✅ All sweeps complete. Check your logs in:")
print(f"  -> {log_csv_path}")
