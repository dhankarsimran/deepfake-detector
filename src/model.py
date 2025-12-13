import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        # input: [3,224,224]
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # [16,224,224]
        self.pool = nn.MaxPool2d(2,2)                # [16,112,112]
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # [32,112,112]
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # [64,112,112]

        self.fc1 = nn.Linear(64*28*28, 128)
        self.fc2 = nn.Linear(128, 2)  # output: 2 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [16,112,112]
        x = self.pool(F.relu(self.conv2(x)))  # [32,56,56]
        x = self.pool(F.relu(self.conv3(x)))  # [64,28,28]
        x = x.view(-1, 64*28*28)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
