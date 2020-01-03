# https://github.com/ikostrikov/pytorch-a3c/blob/48d95844755e2c3e2c7e48bbd1a7141f7212b63f/train.py#L9 for inspiration
# https://github.com/muupan/async-rl/blob/master/a3c.py to verify loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from neurips2019.agents.agent import Agent
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from neurips2019.util.utils import share_weights, share_gradients
from neurips2019.util.Logger import LogEntry, LogType
from random import random, choice
import os


class Worker(Agent, mp.Process):
# Instances of this class are created as separate processes to train the "main" a3c agent
# extends the Agent interface as well as the pyTorch multiprocessing process class

    def __init__(self, logq:mp.Queue, shared_policy, shared_value, shared_policy_optim, shared_value_optim, global_counter, policynetfunc, valuenetfunc, tmax, expl_policy, env_factory, actions, idx, grad_clip=40, gamma=0.99):
        self.env = env_factory.get_instance()
        self.name = f"worker-{idx:02d}"
        self.idx = idx
        self.epsilon = expl_policy
        self.actions = actions # save possible actions
        self.policynet = policynetfunc()
        self.valuenet = valuenetfunc()
        self.NLL = nn.NLLLoss()
        self.grad_clip = grad_clip
        self.tmax = tmax # maximum lookahead
        self.gamma = gamma # discount value
        self.shared_policy = shared_policy
        self.shared_value = shared_value
        self.global_counter = global_counter
        self.shared_policy_optim = shared_policy_optim
        self.shared_value_optim = shared_value_optim

        # copy weights from shared network
        share_weights(self.shared_policy, self.policynet)
        share_weights(self.shared_value, self.valuenet)

        self.logq = logq


    def action(self, state):
        # performs action according to policy, or at random with probability determined by epsilon greedy strategy
        state = torch.FloatTensor(state)
        policy = self.policynet(state)
        with torch.no_grad(): # only save gradient information when calculating the loss TODO: possible source of screwups
            eps = self.epsilon(self.global_counter.value)
            if random() < eps:
                action = choice(self.actions)
            else:
                probs = F.softmax(policy, dim=0).data.numpy()
                probs /= sum(probs) # ensure probs sum to 1
                # idx = np.argmax(probs)
                action = np.random.choice(self.actions, size=None, replace=False, p=probs) # choose action with probability according to policy
                # action = self.actions[idx]

        return policy, action

    def train(self, Tmax, return_dict, clip_grads=True, render=False):
        # train loop: pulls current shared network, and performs actions in its own environment. Computes the gradients for its own networks and pushes them to the shared network which is then updated. Length of training is determined by a global counter that is incremented by all worker processes
        print(f"{self.name}: Training started")
        value_losses = list()
        policy_losses = list()
        reward_eps = list()
        reward_ep = 0
        state = self.env.reset() # reset environment
        done = False

        # repeat until maximum number of episodes is reached
        while self.global_counter.value < Tmax:
            policy_loss = torch.Tensor([0])
            value_loss = torch.Tensor([0])
            # self.policy_optim.zero_grad() # workers should not need their own optimizers
            # self.theta_v_optim.zero_grad()

            # copy weights from shared net
            share_weights(self.shared_policy, self.policynet)
            share_weights(self.shared_value, self.valuenet)

            states = list()
            actions = list()
            rewards = list()
            # compute next tmax steps, break if episode has ended
            for tstep in range(self.tmax):
                # perform action according to policy
                states.append(state)
                policy, action = self.action(state)
                state, reward, done = self.env.step(action)
                actions.append(action)
                rewards.append(reward)
                reward_ep += reward

                if done: # stop early if we reach a terminal state
                    self.logq.put(LogEntry(LogType.SCALAR, f"reward/{self.name}", reward_ep, self.global_counter.value, {}))
                    state = self.env.reset()
                    reward_eps.append(reward_ep)
                    # increment global episode counter
                    with self.global_counter.get_lock():
                        self.global_counter.value += 1
                        if self.global_counter.value % 100 == 0 and self.global_counter.value > 0:
                            print(f"{self.name}:")
                            print(f">> Global Counter: {self.global_counter.value}")
                            print(f">> Current score: {reward_ep}")
                            print(f">> Last 100 mean score: {np.mean(reward_eps[-100:])}")
                            print(f">> Epsilon: {self.epsilon(self.global_counter.value)}")
                            self.env.render = render # render next episode
                        else:
                            self.env.close_window() # closes window after episode is finished
                            self.env.render = False

                    reward_ep = 0
                    break

            if done:
                R = 0
            else:
                R = self._get_value(state) # bootstrap reward from value of last known state

            policy_loss, value_loss, entropy = self.calc_loss(states, actions, rewards, R)
            pl = policy_loss.detach().numpy()[0]
            policy_losses.append(pl)
            vl = value_loss.detach().numpy()[0]
            e = entropy.detach().numpy()[0]
            value_losses.append(vl)
            self.logq.put(LogEntry(LogType.SCALAR, f"policy-loss/{self.name}", pl, self.global_counter.value, {}))
            self.logq.put(LogEntry(LogType.SCALAR, f"value-loss/{self.name}", vl, self.global_counter.value, {}))
            self.logq.put(LogEntry(LogType.SCALAR, f"entropy/{self.name}", e, self.global_counter.value, {}))

            # compute gradients and update shared network
            policy_loss.backward(retain_graph=True) # retain graph as it is needed to backpropagate value_loss as well
            value_loss.backward(retain_graph=False) # now reset the graph to avoid accumulation over multiple iterations

            # clip gradients
            if clip_grads:
                self._clip_gradients()

            # make sure agents do not override each others gradients
            # TODO maybe this is not needed
            #with self.a3c_instance.lock: # at the moment a lock is acquired before workers update the shared net to avoid overriding gradients. For the agent to be truly 'asynchronous' this lock should be removed
            #   pass
            share_gradients(self.valuenet, self.shared_value)
            share_gradients(self.policynet, self.shared_policy)
            self.update_shared_nets()

        #print(f"storing results to {self.idx}-policyloss and {self.idx}-valueloss")
        return_dict[f"{self.idx}-policyloss"] = policy_losses
        return_dict[f"{self.idx}-valueloss"] = value_losses
        return_dict[f"{self.idx}-reward_ep"] = reward_eps

    def _get_value(self, state):
        state = torch.FloatTensor(state)
        value = self.valuenet(state)
        return value

    def _get_log_policy(self, current_state, current_action):
        # returns the log policy of the action taken
        current_state = torch.FloatTensor(current_state)
        current_action = torch.LongTensor([self.actions.index(current_action)]) # convert current_action to tensor
        policy = self.policynet(current_state)
        policy = F.log_softmax(policy, dim=0)
        policy_action = torch.index_select(policy, dim=0, index=current_action)
        return policy_action

    def _get_entropy(self, current_state):
        """ compute entropy of policy at current state

        Args:
            current_state: the current state of the environment

        Returns:
            entropy: entropy value of the current state
        """
        # TODO technically entropy should use log base 2
        current_state = torch.FloatTensor(current_state)
        logit_policy = self.policynet(current_state)
        policy = F.softmax(logit_policy, dim=0)
        log_policy = F.log_softmax(logit_policy, dim=0)
        entropy = -torch.dot(policy, log_policy)
        return entropy

    def calc_loss(self, states, actions, rewards, R, entropy=True):
        # TODO include entropy?
        # compute policy value of action in state
        n_steps = len(rewards)
        policy_loss = torch.Tensor([0])
        value_loss = torch.Tensor([0])
        entropy = torch.Tensor([0])
        for t in range(n_steps-1,-1,-1): # traverse backwards through time
            R = rewards[t] + self.gamma * R
            # calculate policy value at timestep t
            log_policy_t = self._get_log_policy(states[t], actions[t])
            # compute value function at timestep t
            value_t = self._get_value(states[t])
            advantage = (R - value_t)
            policy_loss -= log_policy_t * advantage # TODO -= or +=?
            entropy += self._get_entropy(states[t])
            value_loss += advantage**2
        if entropy:
            policy_loss += entropy * 10 # TODO -=?, TODO weight entropy differently?
        policy_loss = policy_loss ** 2 # non-negative only
        return policy_loss, value_loss, entropy

    def evaluate(self):
        print("Do not evaluate worker instances directly!")

    def _clip_gradients(self):
        clip_grad_norm_(self.policynet.parameters(),  self.grad_clip)
        clip_grad_norm_(self.valuenet.parameters(), self.grad_clip)

    def update_shared_nets(self):
        self.shared_policy_optim.step()
        self.shared_value_optim.step()
        self.shared_policy_optim.zero_grad()
        self.shared_value_optim.zero_grad()
