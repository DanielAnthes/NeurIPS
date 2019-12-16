import torch
import torch.nn as nn
from torch.optim import SGD
from torch.multiprocessing import Lock
from agent import Agent
from Networks import Net
from Worker import Worker

class A3CAgent(Agent):
    def __init__(self, tmax):
        # TODO correctly initialize things
        self.policynet = Net(4, 10, 2) # parameters of policy network
        self.valuenet = Net(4, 10, 2) # parameters for value network
        self.tmax = tmax # maximum lookahead
        self.policy_optim = SGD(self.policynet.parameters())
        self.value_optim = SGD(self.valuenet.parameters())
        self.global_counter = 0
        self.lock = Lock()

    def train(self,Tmax, num_processes):

        self.global_counter = 0
        # repeat for training iterations
        model = Worker(self, 10 

    def update_networks(self):
        self.policy_optim.step()
        self.value_optim.step()
