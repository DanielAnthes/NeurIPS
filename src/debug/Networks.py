from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as xavier, kaiming_normal_ as he
import torchvision.models as models

class Net(nn.Module):
    """
    Implements a simple fully connected network
    """
    def __init__(self, num_in, num_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_in, num_out)

    def forward(self, x):
        return F.relu(self.fc1(x))


class WideNet(nn.Module):
    """
    Implements a simple fully connected network with a variable number > 1 of hidden layers
    """
    def __init__(self, num_in, hidden, num_out, act_func=nn.ReLU):
        super(WideNet, self).__init__()
        if not type(hidden) == list:
            hidden = [hidden]

        layers = OrderedDict()
        m = nn.Linear(num_in, hidden[0])
        xavier(m.weight)
        layers["Input"] = m
        layers["Input_Act"] = act_func()

        if len(hidden) > 1:
            for idx in range(1, len(hidden)):
                m = nn.Linear(hidden[idx-1], hidden[idx])
                xavier(m.weight)
                layers[f"Hidden-{idx}"] = m
                layers[f"Hidden-{idx}-Act"] = act_func()

        layers["Out"] = nn.Linear(hidden[-1], num_out)
        self.net = nn.Sequential(layers)

    def forward(self, x):
        return self.net.forward(x)


class CNN(nn.Module):
    ''' Class implementing the COnvolutional Network given in the DQN tutorial on the pytorch website:
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    '''

    def __init__(self, channels_in, out_features):
        super(CNN, self).__init__()
        self.net = nn.Sequential()
        # ToDo: layer normalization
        # add padding of 1
        m = nn.Conv2d(channels_in, 16, kernel_size=3, stride=2, padding=1)

        he(m.weight)
        self.net.add_module("Conv_1", m)
        # self.net.add_module("BN_1", nn.BatchNorm2d(16))
        self.net.add_module("Act_1", nn.ReLU())
        m = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        he(m.weight)
        self.net.add_module("Conv_2", m)
        # self.net.add_module("BN_2", nn.BatchNorm2d(32))
        self.net.add_module("Act_2", nn.ReLU())
        # self.net.add_module("Flatten", Flatten())
        # ToDo: add 3rd CNN layer
        m = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        he(m.weight)
        self.net.add_module("Conv_3", m)
        self.net.add_module("Act_3", nn.ReLU())
        # self.net.add_module("BN_3", nn.BatchNorm2d(64))
        # ToDo:  global avg pooling -> 64x1x1
        self.net.add_module("Global_Avg_Pool", nn.AvgPool2d(kernel_size=8, stride=0, padding=0))
        # cnn from 64x1x1, kernel_size=1
        m = nn.Conv2d(64, out_features, kernel_size=1)
        xavier(m.weight)
        self.net.add_module("Readout_Conv", m)
        self.net.add_module("Readout_Act", nn.ReLU())

        # m = nn.GRUCell(3600, 3600)
        # xavier(m.weight_ih)
        # xavier(m.weight_hh)
        # self.memory = m

        # m = nn.Linear(3600, outputs)
        # xavier(m.weight)
        # self.net.add_module("Readout", m)
        # self.net.add_module("Act_3", nn.ReLU())

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.net(x).squeeze()


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class PretrainedResNet(nn.Module):
    def __init__(self, outputs):
        super(PretrainedResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False # freeze parameters
        self.net = nn.Sequential(*list(resnet.children())[:-14]) # only take first 4 conv layers
        self.net.add_module("Flatten", Flatten())
        self.net.add_module("Readout", nn.Linear(12288, outputs))

    def forward(self, x):
        return self.net(x)

class PretrainedSqueezeNet(nn.Module):
    def __init__(self, outputs):
        super(PretrainedSqueezeNet, self).__init__()
        squeezenet = models.squeezenet1_1(pretrained=True)
        for param in squeezenet.parameters():
            param.requires_grad = False # freeze parameters
        self.net = nn.Sequential(*list(squeezenet.children())[:-1]) # exclude output layer
        self.net.add_module("Flatten", Flatten())
        self.net.add_module("Readout", nn.Linear(2048, outputs))

    def forward(self, x):
        return self.net(x)

class GermainNet(nn.Module):
    """adapted from https://github.com/germain-hug/Deep-RL-Keras"""
    def __init__(self):
        super(GermainNet, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module("Conv1", nn.Conv2d(3, 32, 3))
        self.net.add_module("ActC1", nn.ReLU())
        self.net.add_module("Pool1", nn.MaxPool2d(2))
        self.net.add_module("Conv2", nn.Conv2d(32, 32, 3))
        self.net.add_module("ActC2", nn.ReLU())
        self.net.add_module("Pool2", nn.MaxPool2d(2))
        self.net.add_module("Flatten", Flatten())

    def forward(self, x):
        return self.net(x)

class GermainActor(nn.Module):
    """adapted from https://github.com/germain-hug/Deep-RL-Keras"""
    def __init__(self, actionspace):
        super(GermainActor, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module("FFN", nn.Linear(6272, 128))
        self.net.add_module("Act1", nn.ReLU())
        self.net.add_module("Readout", nn.Linear(128, actionspace))
        self.net.add_module("Readout_Act", nn.Softmax())

    def forward(self, x):
        return self.net(x)

class GermainCritic(nn.Module):
    """adapted from https://github.com/germain-hug/Deep-RL-Keras"""
    def __init__(self):
        super(GermainCritic, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module("FFN", nn.Linear(6272, 128))
        self.net.add_module("Act1", nn.ReLU())
        self.net.add_module("Readout", nn.Linear(128, 1))
        
    def forward(self, x):
        return self.net(x)