import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkCIFAR(nn.Module):

    def __init__(self, C):
        super(NetworkCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)         # Conv2d(in_channels, out_channels, kernel_size, ...)
        self.pool = nn.MaxPool2d(2, 2)          # MaxPool2d(kernel_size, stride=None, ...)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # Linear(in_features, hidden_features, ...) : y = x * A^T + b
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)            # Linear(in_features, out_features, ...) : y = x * A^T + b

    def forward(self, x):
        logits_aux = None
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits, logits_aux