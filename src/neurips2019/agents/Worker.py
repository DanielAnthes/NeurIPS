# https://github.com/ikostrikov/pytorch-a3c/blob/48d95844755e2c3e2c7e48bbd1a7141f7212b63f/train.py#L9 for inspiration
import torch
from torch.optim import SGD
import torch.nn.functional as F
import numpy as np
import gym
from Networks import Net # TODO implement better network
from agent import Agent
import torch.multiprocessing as mp
from utils import share_weights, share_gradients

class Worker(Agent, mp.Process):

    def __init__(self, a3c_instance, tmax, env_factory, actions, name):
        self.env = env_factory.get_instance()
        self.name = name
        self.actions = actions # save possible actions
        self.policynet = Net(4, 10, 2) # policy network
        self.valuenet = Net(4, 10, 2) # value function network

        # copy weights from shared net
        share_weights(a3c_instance.policynet, self.policynet)
        share_weights(a3c_instance.valuenet, self.valuenet)

        self.tmax = tmax # maximum lookahead
        self.policy_optim = SGD(self.policynet.parameters())
        self.theta_v_optim = SGD(self.valuenet.parameters())
        self.a3c_instance = a3c_instance # store reference to main agent
        self.gamma = 0.99 # discount value

    def action(self, state):
        # performs action according to policy
        # action is picked with probability proportional to policy values
        policy = self.policynet(state)
        probs = F.softmax(policy, dim=0).data.numpy()
        probs /= sum(probs)  # make sure vector sums to 1
        action = np.random.choice(self.actions, size=None, replace=False, p=probs)
        return policy, action

    def train(self, Tmax):
        state = self.env.reset() # reset environment

        # save states, actions and rewards
        states = list()
        actions = list()
        rewards = list()

        # repeat until maximum number of steps is reached
        while self.a3c_instance.global_counter < Tmax:
            # reset gradients
            self.policy_optim.zero_grad()
            self.theta_v_optim.zero_grad()

            # copy weights from shared net
            share_weights(self.a3c_instance.policynet, self.policynet)
            share_weights(self.a3c_instance.valuenet, self.valuenet)

            # compute next tmax steps, break if episode has ended
            for tstep in range(self.tmax):

                # perform action according to policy
                states.append(state)
                policy, action = self.action(state)
                state, reward, done = self.env.step(action)
                actions.append(action)
                rewards.append(reward)

                if done: # stop early if we reach a terminal state
                    # increment global episode counter
                    with self.a3c_instance.global_counter.get_lock():
                        self.a3c_instance.global_counter.value += 1
                        R = 0
                    break
            # initialize R
            if not done:
                R = self._get_value(state)
            policy_loss, value_loss = self.calc_loss(self, states, actions, rewards)

            # compute gradients and update shared network
            policy_loss.backward()
            value_loss.backward()

            # make sure agents do not override each others gradients
            # TODO maybe this is not needed
            with self.a3c_instance.lock:
                share_gradients(self.valuenet, self.a3c_instance.valuenet)
                share_gradients(self.policynet, self.a3c_instance.policynet)
                self.a3c_instance.update_networks()

    def _get_value(self, state):
        state = torch.from_numpy(state)
        value = self.valuenet(state)
        return value

    def _get_policy(self, current_state, current_action):
        current_state = torch.from_numpy()
        current_action = torch.from_numpy(current_action)
        policy = self.policynet(current_state)
        policy_action = torch.index_select(policy, dim=0, index=current_action)
        return policy_action

    def calc_loss(self, states, actions, rewards, R):
        # TODO include entropy?
        # compute policy value of action in state
        n_steps = length(rewards)
        policy_loss = 0
        value_loss = 0

        for t in range(n_steps-1,-1,-1): # traverse backwards through time
            R = rewards[t] + self.gamma * R
            # calculate policy value at timestep t
            policy_t = self._get_policy(states[t], actions[t])
            # compute value function at timestep t
            value_t = self._get_value(states[t])

            policy_loss += torch.log(policy_t) * (R - value_t)
            value_loss += (R - value_t)**2

        return policy_loss, value_loss


    def evaluate(self):
        print("Do not evaluate worker instances directly!")
