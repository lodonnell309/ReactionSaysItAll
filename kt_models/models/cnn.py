import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, # RGB
                              out_channels=48,
                              kernel_size=7, # square kernel
                              stride=2,
                              padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, # square kernel
                                 stride=2)
        self.fc = nn.Linear(in_features = 4800, # hard-coded based on the data shape
                            out_features = 7,
                            bias = True)

    def forward(self, x):
        # print(x.shape) # [128, 1, 48, 48]

        x = self.conv(x) # no need to reshape for convolution
        # print(x.shape) # [128, 48, 21, 21]

        x = self.relu(x)
        # print(x.shape) # [128, 48, 21, 21]

        x = self.pool(x)
        # print(x.shape) # [128, 48, 10, 10]

        # reshape for linear
        x = torch.reshape(x, (x.shape[0], -1))  # here, -1 is inferred from the remaining dimensions
        # print(x.shape) # [128, 4800]

        outs = self.fc(x)
        # print(outs.shape) # [128, 7]

        return outs
