import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # first conv block
        self.conv = nn.Conv2d(in_channels=1, # RGB
                              out_channels=16,
                              kernel_size=4, # square kernel
                              stride=1,
                              padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, # square kernel
                                 stride=2)
        # second conv block
        self.conv2 = nn.Conv2d(in_channels=16, # RGB
                              out_channels=32,
                              kernel_size=4, # square kernel
                              stride=1,
                              padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, # square kernel
                                 stride=2)
        self.fc = nn.Linear(in_features = 3872, # hard-coded based on the data shape
                            out_features = 1800,
                            bias = True)
        self.fc2 = nn.Linear(in_features = 1800, # hard-coded based on the data shape
                            out_features = 7,
                            bias = True)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        # print(x.shape) # [128, 1, 48, 48]

        x = self.conv(x) # no need to reshape for convolution
        # print(x.shape) # [128, 16, 47, 47]
        x = self.relu(x)
        # print(x.shape) # [128, 16, 47, 47]
        x = self.pool(x)
        # print(x.shape) # [128, 16, 23, 23]

        x = self.conv2(x) # no need to reshape for convolution
        # print(x.shape) # [128, 32, 22, 22]
        x = self.relu2(x)
        # print(x.shape) # [128, 32, 22, 22]
        x = self.pool2(x)
        # print(x.shape) # [128, 32, 11, 11]

        x = self.drop(x)

        # reshape for linear
        x = torch.reshape(x, (x.shape[0], -1))  # here, -1 is inferred from the remaining dimensions
        # print(x.shape) # [128, 3872]

        x = self.fc(x)
        # print(outs.shape) # [128, 1800]

        outs = self.fc2(x)
        # print(outs.shape) # [128, 7]

        return outs
