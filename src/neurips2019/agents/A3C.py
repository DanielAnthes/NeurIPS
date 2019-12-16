import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.multiprocessing import Lock, Value, Process
from neurips2019.agents.agent import Agent
from neurips2019.agents.Networks import Net
from neurips2019.agents.Worker import Worker
import numpy as np

class A3CAgent(Agent):
    def __init__(self, tmax, env_factory, actions):
        # TODO correctly initialize things
        self.policynet = Net(4, 10, 2) # parameters of policy network
        self.valuenet = Net(4, 10, 1) # parameters for value network
        self.tmax = tmax # maximum lookahead
        self.policy_optim = SGD(self.policynet.parameters(), lr=0.01)
        self.value_optim = SGD(self.valuenet.parameters(), lr=0.01)
        self.global_counter = Value('i', 0) # global episode counter
        self.env_factory = env_factory
        self.actions = actions
        self.lock = Lock()

    def train(self, Tmax, num_processes):
        # repeat for training iterations
        processes = list()
        for i in range(num_processes):
            worker = Worker(self, 10, self.env_factory, self.actions, f"worker-{i}")
            processes.append(Process(target=worker.train, args=(Tmax,)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def update_networks(self):
        self.policy_optim.step()
        self.value_optim.step()

    def action(self, state):
        # performs action according to policy
        # action is picked with probability proportional to policy values
        policy = self.policynet(state)
        probs = F.softmax(policy, dim=0).data.numpy()
        probs /= sum(probs)  # make sure vector sums to 1
        action = np.random.choice(self.actions, size=None, replace=False, p=probs)
        return policy, action

    def calc_loss(self):
        print("loss should be calculated in the Workers")
        return None

    def evaluate(self):
        # TODO
        pass
