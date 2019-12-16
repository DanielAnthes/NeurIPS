import torch
import torch.nn as nn
from torch.optim import SGD
from torch.multiprocessing import Lock, Value, Process
from agent import Agent
from Networks import Net
from Worker import Worker

class A3CAgent(Agent):
    def __init__(self, tmax, env_factory, actions):
        # TODO correctly initialize things
        self.policynet = Net(4, 10, 2) # parameters of policy network
        self.valuenet = Net(4, 10, 2) # parameters for value network
        self.tmax = tmax # maximum lookahead
        self.policy_optim = SGD(self.policynet.parameters())
        self.value_optim = SGD(self.valuenet.parameters())
        self.global_counter = Value('i', 0) # global episode counter
        self.env_factory = env_factory
        self.actions = actions
        self.lock = Lock()

    def train(self,Tmax, num_processes):
        self.global_counter = 0
        # repeat for training iterations
        processes = list()
        for i in range(num_processes):
            worker = Worker(self, 10, self.env_factory, self.actions, f"worker-{i}")
            processes.append(Process(target=worker.train, args={'Tmax': self.global_counter}))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def update_networks(self):
        self.policy_optim.step()
        self.value_optim.step()
