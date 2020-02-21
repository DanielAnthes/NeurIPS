from Logger import LogEntry, LogType
import torch.multiprocessing as mp
import Networks as N
import gym
import torch
import torch.nn.functional as F
from random import random, choice
import numpy as np

class Worker(mp.Process):

    def __init__(self, global_counter, global_max_episodes, shared_value, shared_policy, shared_optim, log_queue, name):

        # networks
        self.valuenet = N.WideNet(4, 16, 1)
        self.policynet = N.WideNet(4, 16, 2)

        self.shared_value = shared_value
        self.shared_policy = shared_policy
        self.shared_optim = shared_optim

        self.global_counter = global_counter
        self.global_max_episodes = global_max_episodes

        self.logq= log_queue

        # parameters
        self.lookahead = 10
        self.gamma = 0.95
        self.actions = [0, 1]
        self.name = name

    def train(self):
        print("Worker started training")
        value_losses = list()
        policy_losses = list()
        reward_eps = list()
        reward_ep = 0
        env = gym.make('CartPole-v1')
        state = env.reset()
        done = False

        # repeat until maximum number of episodes is reached
        while self.global_counter.value < self.global_max_episodes:
            policy_loss = torch.Tensor([0])
            value_loss = torch.Tensor([0])

            # copy weights from shared net
            self.share_weights(self.shared_policy, self.policynet)
            self.share_weights(self.shared_value, self.valuenet)

            states = list()
            actions = list()
            rewards = list()

            for t in range(self.lookahead):
                states.append(state)
                policy, action = self.action(state)
                state,reward, done, _ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                reward_ep += reward

                if done:
                    state = env.reset()
                    reward_eps.append(reward_ep)
                    with self.global_counter.get_lock():
                        self.global_counter.value += 1
                    self.logq.put(LogEntry(LogType.SCALAR, f"reward/{self.name}", reward_ep, self.global_counter.value, {}))
                    reward_ep = 0

            # compute loss over last "lookahead"
            if done:
                R = 0
            else:
                R = self.valuenet(torch.FloatTensor(state))

            n_steps = len(rewards)
            policy_loss = 0
            value_loss = 0

            for t in range(n_steps-1,-1,-1): # traverse backwards through states
                R = rewards[t] + self.gamma * R
                current_state = torch.FloatTensor(states[t])
                current_action = torch.LongTensor([actions[t]])
                policy = self.policynet(current_state)
                policy = F.log_softmax(policy, dim=0)
                log_policy_t = torch.index_select(policy, dim=0, index=current_action) # policy value of action that was performed
                value_t = self.valuenet(current_state)
                advantage = R -value_t
                policy_loss -= log_policy_t * advantage
                value_loss += advantage**2
            loss = value_loss + policy_loss
            self.logq.put(LogEntry(LogType.SCALAR, f"loss/{self.name}", loss.detach(), self.global_counter.value, {}))
            loss.backward()

            # push gradients to shared network
            self.share_gradients(self.valuenet, self.shared_value)
            self.share_gradients(self.policynet, self.shared_policy)

            # optimize shared nets
            self.shared_optim.step()
            self.shared_optim.zero_grad()

        # close environment after training
        env.close()

    def action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            policy = self.policynet(state)
            if random() < self.epsilon():
                action = choice(self.actions) # random action
            else:
                probs = F.softmax(policy, dim=0).data.numpy()
                probs /= sum(probs)
                action = np.random.choice(self.actions, size=None, replace=False, p=probs)
        return policy, action


    def epsilon(self):
        eps = 1 - (self.global_counter.value / self.global_max_episodes) # linearly decreasing epsilon as a function of training percentage completed
        if self.name == "Worker-0":
            self.logq.put(LogEntry(LogType.SCALAR, "epsilon", eps, self.global_counter.value, {}))
        return eps

    def share_weights(self, from_net, to_net):
        '''takes two pytorch networks and copies weights from the first to the second network'''
        params = from_net.state_dict()
        to_net.load_state_dict(params)



    def share_gradients(self, from_net, to_net):
        for from_param, to_param in zip(from_net.parameters(), to_net.parameters()):
            to_param._grad = from_param.grad

