import matplotlib.pyplot as plt
import tqdm
import numpy as np
import random
import socket
import struct
from chainer import Chain
import chainer.links as L
import chainer.functions as F
from chainer.optimizers import Adam
from chainer import Variable

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


class RandomAgent:
    def __init__(self):
        pass

    def step(self, reward, state):
        return random.randint(0, 1)


class MLP(Chain):
    """Multilayer perceptron"""

    def __init__(self, n_output=1, n_hidden=5):
        super(MLP, self).__init__(l1=L.Linear(None, n_hidden), l2=L.Linear(n_hidden, n_output))

    def __call__(self, x):
        return self.l2(F.relu(self.l1(x)))



# A skeleton for the REINFORCEAgent is given. Implement the compute_loss and compute_score functions.

class REINFORCEAgent(object):
    """Agent trained using REINFORCE"""

    def __init__(self, model, optimizer=Adam()):
        self.model = model

        self.optimizer = optimizer
        self.optimizer.setup(self.model)

        # monitor score and reward
        self.rewards  = []
        self.policies = []
        self.scores   = []

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

        # cost to go
        Qhat = 0

        loss = 0
        for t in range(len(self.rewards) - 1, -1, -1):

            Qhat = self.rewards.pop() + Qhat

            _ss = F.squeeze(self.scores.pop(), axis=1) * Qhat

            if _ss.size > 1:
                _ss = F.sum(_ss, axis=0)
            loss += F.squeeze(_ss)

        return loss


    def compute_score(self, action, policy):
        """
        Computes score

        Args:
            action (int):
            policy:

        Returns:
            score
        """
        # computes log softmax of policy and selects the value for the action that was actually performed
        score = F.select_item(F.log_softmax(policy), Variable(action))
        if score.ndim == 1:
            score = F.expand_dims(score, axis=1)
        return score



'''
class DQNAgent(object):
    """Agent trained using DQN"""

    def __init__(self, qnet, qnet_hat, optimizer=Adam, buff_size=1000, eta=1.0):
        # create qnets
        self.qnet       = qnet
        self.qnet_hat   = qnet_hat
        self.optimizer = optimizer(params=qnet.parameters())

        # initialize experience buffer
        buffer_type = np.dtype([('s','f8',4),('a','i4'),('r','f8'),('sprime','f8',4)])
        self.buffer = np.zeros(buff_size, dtype=buffer_type)

        # set exploration factor
        self.eta = eta

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

'''
