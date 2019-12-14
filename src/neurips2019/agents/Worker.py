import torch
from torch.optim import SGD
import numpy as np
import gym
from Networks import Net # TODO implement better network
from agent import Agent

class Worker(Agent):

    def __init__(self, a3c_instance, tmax, env):
        # TODO correctly initialize things with torch
        self.env = env
        self.policynet = Net(4, 10, 2) # policy network
        self.valuenet = Net(4, 10, 2) # value function network
        self.tmax = tmax # maximum lookahead
        self.policy_optim = SGD(self.policynet.parameters())
        self.theta_v_optim = SGD(self.valuenet.parameters())
        self.a3c_instance = a3c_instance # store reference to main agent
        self.gamma = 0.99 # discount value

    def _synchronize_weights(self, theta, theta_v):
        # TODO use pytorch function for copying weights here?
        self.theta = theta
        self.theta_v = theta_v

    def action(self):
        # TODO
        pass

    def train(self, Tmax):
        # TODO
        # initialize thread step counter
        t = 0
        # repeat until maximum number of steps is reached
        for i in range(Tmax):
            # reset gradients
            self.policy_optim.zero_grad()
            self.theta_v_optim.zero_grad()
            self._synchronize_weights(self.a3c_instance.get_theta(), self.a3c_instance.get_theta_v())
            # compute next tmax steps, break if episode has ended
            for tstep in range(self.tmax):
                # perform action according to policy
                # TODO
                pass

    def _get_value(self, state):
        # TODO
        return None

    def _get_policy(self, current_state, current_action):
        # TODO
        return None

    def calc_loss(self, states, actions, rewards):
        # TODO include entropy?
        # compute policy value of action in state
        current_state = states[0]
        current_action = actions[0]
        policy_t = self._get_policy(current_state, current_action)
        # compute value function at timestep t
        value_t = self._get_value(current_state)
        # compute advantage
        advantage = 0
        for k in range(len(states)):
            advantage += self.gamma**k * rewards[k] + self.gamma**k * self._get_value(states[k]) - value_t
        return torch.log(policy_t) * advantage

    def evaluate(self):
        print("Do not evaluate worker instances directly!")
