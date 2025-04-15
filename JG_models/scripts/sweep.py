# scripts/sweep.py

import itertools, os, yaml, subprocess
from datetime import datetime
import torch


# Define the sweep space
search_space = {
    "dropout": [0.2, 0.3, 0.5],
    "lr": [0.001],
    "batch_size": [64, 128, 256],
}

base_config_path = "configs/base.yaml"
log_dir = "outputs/logs"
os.makedirs(log_dir, exist_ok=True)

# Generate param combinations
keys, values = zip(*search_space.items())
sweep_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

for i, params in enumerate(sweep_configs):
    print(f"\n🚀 Running sweep {i+1}/{len(sweep_configs)}: {params}")

    # Load base config
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Inject sweep params
    config["network"]["dropout"] = params["dropout"]
    config["train"]["lr"] = params["lr"]
    config["data"]["batch_size"] = params["batch_size"]

    # Create unique config name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"sweep_{i+1:02d}_{timestamp}"
    sweep_cfg_path = f"configs/sweeps/{run_id}.yaml"
    os.makedirs(os.path.dirname(sweep_cfg_path), exist_ok=True)

    # Write out config file
    with open(sweep_cfg_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    subprocess.run(["python", "train.py", "--config_file", sweep_cfg_path])
    torch.cuda.empty_cache()

