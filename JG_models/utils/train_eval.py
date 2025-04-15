import torch, os
import csv
from datetime import datetime

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.sum = self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_params(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total}")

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean()

def log_experiment(output_path, config, train_loss, train_acc, val_loss, val_acc):
    log_file = os.path.join(output_path, "experiment_log.csv")
    is_new_file = not os.path.exists(log_file)

    config_str = (
        f"dropout={config.network.dropout}, "
        f"lr={config.train.lr}, "
        f"batch_size={config.data.batch_size}, "
        f"num_epochs={config.train.num_epochs}, "
        f"use_class_weights={config.loss.use_class_weights}"
    )

    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if is_new_file:
            writer.writerow(["timestamp", "train loss", "train acc", "val loss", "val acc", "config"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(train_loss, 4),
            round(train_acc, 4),
            round(val_loss, 4),
            round(val_acc, 4),
            config_str
        ])
