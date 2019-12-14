import torch
import torch.nn as nn
import gym

class A3CAgent:
    def __init__(self, tmax):
        # TODO correctly initialize things
        self.T = 0 # global shared counter
        self.theta = None # parameters of policy network
        self.theta_v = None # parameters for value network
        self.tmax = tmax # maximum lookahead
        self.theta_optim = None
        self.theta_v_optim = None

    def train(self,Tmax):
        # repeat for training iterations
        for i in range(Tmax):
            # reset gradients
            self
