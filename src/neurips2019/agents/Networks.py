from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Implements a simple fully connected network 
    """
    def __init__(self, num_in, num_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_in, num_out)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))


class WideNet(nn.Module):
    """
    Implements a simple fully connected network with a variable number > 1 of hidden layers
    """
    def __init__(self, num_in, hidden, num_out, act_func=nn.ReLU):
        super(WideNet, self).__init__()
        if not type(hidden) == list:
            hidden = [hidden]

        layers = OrderedDict()
        layers["Input"] = nn.Linear(num_in, hidden[0])
        layers["Input_Act"] = act_func()

        if len(hidden) > 1:
            for idx in range(1, len(hidden)):
                layers[f"Hidden-{idx}"] = nn.Linear(hidden[idx-1], hidden[idx])
                layers[f"Hidden-{idx}-Act"] = act_func()

        layers["Out"] = nn.Linear(hidden[-1], num_out)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net.forward(x)


class CNN(nn.Module):
    ''' Class implementing the COnvolutional Network given in the DQN tutorial on the pytorch website:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''

    def __init__(self, outputs):
        super(CNN, self).__init__()
#        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
#        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
#        self.head = nn.Linear(512, outputs)
        self.net = nn.Sequential()
        self.net.add_module("Conv_1", nn.Conv2d(3, 16, kernel_size=4, stride=2))
#        self.net.add_module("BN_1", nn.BatchNorm2d(16))
        self.net.add_module("Act_1", nn.SELU())
        self.net.add_module("Conv_2", nn.Conv2d(16, 16, kernel_size=4, stride=4))
#        self.net.add_module("BN_2", nn.BatchNorm2d(32))
        self.net.add_module("Act_2", nn.SELU())
        self.net.add_module("Flatten", Flatten())
        self.net.add_module("Readout", nn.Linear(256, outputs))
#        self.net.add_module("Act_3", nn.Tanh())
#        
#        self.conv3 = nn.Conv2d(32, 32, kernel_size=4, stride=1)
#        self.bn3 = nn.BatchNorm2d(32)
        # self.head = nn.Linear(288, outputs) # NOTE This only works for input images of 64x64!

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
#        x = F.elu(self.conv1(x))
#        x = F.elu(self.conv2(x))
#        x = F.elu(self.conv3(x))
#        x = F.elu(self.conv4(x))
#        x = F.relu(self.bn1(self.conv1(x)))
#        x = F.relu(self.bn2(self.conv2(x)))
#        x = F.relu(self.bn3(self.conv3(x)))
#        x = x.view(x.size(0), -1)
#        return self.head(x)
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
