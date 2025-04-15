# main.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from models.cnn_model import EmotionCNN
from utils.train_eval import train, evaluate, save_checkpoint
from utils.dataloaders import get_dataloaders_pt                               
from utils.config_loader import load_config
from utils.device import get_device


def main():
    # Load config
    cfg = load_config()
    print("Config:", cfg)


    # Setup
    torch.backends.cudnn.benchmark = True
    device = get_device()

    train_loader, test_loader = get_dataloaders_pt('data/fer2013_pt', cfg['batch_size'], cfg['num_workers'])

    model = EmotionCNN(num_classes=cfg['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(cfg['num_epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['num_epochs']}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        save_checkpoint(model, os.path.join(cfg['checkpoint_dir'], f"epoch_{epoch+1}.pth"))

    os.makedirs(cfg['log_dir'], exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(cfg['log_dir'], 'loss_curve.png'))

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(cfg['log_dir'], 'accuracy_curve.png'))

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # Optional on Windows
    main()
