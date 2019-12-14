import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_in, num_hidden, num_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_in, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
