
import argparse, yaml
from utils.config_loader import Config
from trainer.trainer_cnn import CNNTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Trainer")
    parser.add_argument('--config_file', type=str, default='configs/base.yaml')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    config = Config(config_dict)
    trainer = CNNTrainer(config)
    trainer.train()
    
