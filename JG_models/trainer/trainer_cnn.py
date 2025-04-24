import os, torch
from torch import nn, optim
from utils.dataset_pt import FER2013TensorDataset
from utils.dataloaders import get_dataloaders_pt
from utils.dataloaders import get_dataloaders_original
from utils.device import get_device
from utils.seed import set_seed
from models.cnn_model import EmotionCNN
from utils.train_eval import AverageMeter, compute_params, save_checkpoint, accuracy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn import functional as F
from utils.train_eval import log_experiment

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class CNNTrainer:
    def __init__(self, config):
        self.cfg = config
        set_seed(self.cfg.seed)
        self.device = get_device()

        self.train_loader, self.test_loader = get_dataloaders_original(
            self.cfg.data.root,
            self.cfg.data.batch_size,
            self.cfg.data.num_workers,
            input_size=self.cfg.network.input_size
        )

        self.model = EmotionCNN(num_classes=self.cfg.network.num_classes, dropout=self.cfg.network.dropout).to(self.device)

        if self.cfg.loss.use_class_weights:
            class_counts = torch.tensor(self.cfg.loss.class_counts, dtype=torch.float32)
            weights = 1.0 / class_counts
            weights = weights / weights.sum()
            weights = weights.to(self.device)
            print("✅ Using class weights:", weights.cpu().numpy())
        else:
            weights = None

        if getattr(self.cfg.loss, 'type', 'cross_entropy') == 'focal':
            print("🔍 Using Focal Loss")
            gamma = getattr(self.cfg.loss, 'gamma', 2.0)
            self.criterion = FocalLoss(alpha=weights, gamma=gamma)
        else:
            print("🔍 Using Cross Entropy Loss")
            self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.train.lr)

        compute_params(self.model)
        os.makedirs(self.cfg.train.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfg.train.log_dir, exist_ok=True)

    def train(self):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = -1
        patience = getattr(self.cfg.train, 'early_stop_patience', 3)
        patience_counter = 0

        for epoch in range(self.cfg.train.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.train.num_epochs}")
            train_loss, train_acc = self._run_epoch(epoch, train=True)
            val_loss, val_acc = self._run_epoch(epoch, train=False)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                save_checkpoint(self.model, os.path.join(self.cfg.train.checkpoint_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                    break

            save_checkpoint(self.model, os.path.join(self.cfg.train.checkpoint_dir, f"epoch_{epoch+1}.pth"))

        log_experiment(
            output_path=self.cfg.train.log_dir,
            config=self.cfg,
            train_loss=train_losses[best_epoch-1],
            train_acc=train_accs[best_epoch-1],
            val_loss=val_losses[best_epoch-1],
            val_acc=val_accs[best_epoch-1]
        )

        print(f"\n✅ Best Val Accuracy: {best_val_acc:.4f} @ epoch {best_epoch}")
        self._plot_curves(train_losses, val_losses, train_accs, val_accs)
        self.evaluate_confusion_matrix()

    def _plot_curves(self, train_losses, val_losses, train_accs, val_accs):
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.train.log_dir, 'loss_curve.png'))
        plt.close()

        plt.figure()
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.train.log_dir, 'accuracy_curve.png'))
        plt.close()

    def _run_epoch(self, epoch, train=True):
        loader = self.train_loader if train else self.test_loader
        self.model.train() if train else self.model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        desc = f"{'Train' if train else 'Val'} Epoch {epoch + 1}"
        loop = tqdm(loader, desc=desc, leave=False)

        with torch.set_grad_enabled(train):
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                acc = accuracy(outputs, labels)
                loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(acc.item(), images.size(0))

                loop.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)

        phase = "Train" if train else "Val"
        print(f"{phase} Loss: {loss_meter.avg:.4f}, {phase} Acc: {acc_meter.avg:.4f}")
        return loss_meter.avg, acc_meter.avg

    def evaluate_confusion_matrix(self):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        cm = confusion_matrix(all_labels, all_preds)
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.cfg.train.log_dir, "confusion_matrix.png"))
        plt.close()
