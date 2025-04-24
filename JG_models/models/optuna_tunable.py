import torch
import torch.nn as nn
import torch.nn.functional as F

class TunableCNN(nn.Module):
    def __init__(self, conv1_out=64, conv2_out=128, conv3_out=256, dropout=0.3, num_classes=7):
        super().__init__()
        neg_slope = 0.01

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, conv1_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_out),
            nn.LeakyReLU(neg_slope),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_out),
            nn.LeakyReLU(neg_slope),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(conv2_out, conv3_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv3_out),
            nn.LeakyReLU(neg_slope),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv3_out, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
