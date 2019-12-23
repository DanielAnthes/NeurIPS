from collections import OrderedDict

import torch.nn as nn


class Net(nn.Module):
    """
    Implements a simple fully connected network with a variable number > 1 of hidden layers
    """
    def __init__(self, num_in, hidden, num_out, act_func=nn.ReLU):
        super(Net, self).__init__()
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
