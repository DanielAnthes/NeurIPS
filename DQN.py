import matplotlib.pyplot as plt
import tqdm
import numpy as np
import random
import socket
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# same environment as last week
class Environment:
    def __init__(self, ip = "127.0.0.1", port = 13000):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip     = ip
        self.port   = port

        self.client.connect((ip, port))

    def reset(self):
        self._send(0, 0)
        return self._receive()

    def step(self, action):
        self._send(action, 1)
        return self._receive()

    def _receive(self):
        data = self.client.recv(19)
        reward = data[0]
        state = [struct.unpack("@f", data[1 + i * 4: 5 + i * 4])[0] for i in range(4)]
        status = [data[17], data[18]]
        return reward, state, status

    def _send(self, action, command):
        self.client.send(bytes([action, command]))


class Net(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_in, n_hidden)  # 6*6 from image dimension
        self.fc2 = nn.Linear(n_hidden, n_out)



    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent(object):
    """Agent trained using DQN"""

    def __init__(self, qnet, qnet_hat, optimizer=Adam):
        self.qnet       = qnet
        self.qnet_hat   = qnet_hat

        self.optimizer = optimizer(params=qnet.parameters())

        # monitor score and reward
        self.rewards    = []
        self.scores     = []


    def step(self, reward, state):

        # linear outputs reflecting the log action probabilities and the value
        policy = self.model(Variable(np.atleast_2d(np.asarray(state, 'float32'))))

        # generate action according to policy
        p = F.softmax(policy).data

        # normalize p in case tiny floating precision problems occur
        row_sums = p.sum(axis=1)
        p /= row_sums[:, np.newaxis]

        action = np.asarray([np.random.choice(p.shape[1], None, True, p[0])])

        return action, policy


    def compute_loss(self):
        """
        Return loss for this episode based on computed scores and accumulated rewards
        """

        pass

    def compute_score(self, action, policy):
        """
        Computes score

        Args:
            action (int):
            policy:

        Returns:
            score
        """

        pass
