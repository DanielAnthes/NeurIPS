from Logger import LogEntry, LogType
import torch.multiprocessing as mp
import Networks as N
import gym
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from random import random, choice
import numpy as np
from utils import resize, get_state

class Worker(mp.Process):

    def __init__(self, global_counter, global_max_episodes, shared_conv, shared_value, shared_policy, shared_optim, log_queue, name, evaluate):

        # networks
        self.convnet = N.CNN(128)
        self.valuenet = N.WideNet(128, 32, 1)
        self.policynet = N.WideNet(128, 32, 2)

        self.shared_value = shared_value
        self.shared_policy = shared_policy
        self.shared_conv = shared_conv
        self.shared_optim = shared_optim
        self.evaluate = evaluate

        self.global_counter = global_counter
        self.global_max_episodes = global_max_episodes

        self.logq= log_queue

        # parameters
        self.lookahead = 10
        self.gamma = 0.95
        self.actions = [0, 1]
        self.name = name
        self.max_norm = 1

    def train(self):
        print("Worker started training")
        reward_eps = list()
        reward_ep = 0
        env = gym.make('CartPole-v1')
        env.reset()
        state = get_state(env)
        done = False

        # repeat until maximum number of episodes is reached
        while self.global_counter.value < self.global_max_episodes:
            policy_loss = torch.Tensor([0])
            value_loss = torch.Tensor([0])

            # copy weights from shared net
            self.share_weights(self.shared_policy, self.policynet)
            self.share_weights(self.shared_value, self.valuenet)
            self.share_weights(self.shared_conv, self.convnet)

            states = list()
            actions = list()
            rewards = list()

            for t in range(self.lookahead):
                states.append(state)
                policy, action = self.action(state)
                _ ,reward, done, _ = env.step(action)
                state = get_state(env)
                actions.append(action)
                rewards.append(reward)
                reward_ep += reward

                if done:
                    env.reset()
                    state = get_state(env)
                    reward_eps.append(reward_ep)
                    with self.global_counter.get_lock():
                        self.global_counter.value += 1
                    self.logq.put(LogEntry(LogType.SCALAR, f"reward/{self.name}", reward_ep, self.global_counter.value, {}))
                    reward_ep = 0

                    if self.global_counter.value % 200 == 0:
                        eval_rewards = self.evaluate(10)
                        print(f"MEAN EVALUATION REWARD: {np.mean(eval_rewards)}")

            # compute loss over last "lookahead"
            if done:
                R = 0
            else:
                representation = self.convnet(torch.FloatTensor(state).unsqueeze(dim=0))
                R = self.valuenet(representation)

            n_steps = len(rewards)
            policy_loss = 0
            value_loss = 0

            for t in range(n_steps-1,-1,-1): # traverse backwards through states
                R = rewards[t] + self.gamma * R
                current_state = torch.FloatTensor(states[t]).unsqueeze(dim=0)
                current_action = torch.LongTensor([actions[t]])
                current_representation = self.convnet(current_state).squeeze(dim=0)
                policy = self.policynet(current_representation)
                policy = F.log_softmax(policy, dim=0)
                log_policy_t = torch.index_select(policy, dim=0, index=current_action) # policy value of action that was performed
                value_t = self.valuenet(current_representation)
                advantage = R -value_t
                policy_loss -= log_policy_t * advantage
                value_loss += advantage**2
            loss = value_loss + policy_loss
            self.logq.put(LogEntry(LogType.SCALAR, f"loss/{self.name}", loss.detach(), self.global_counter.value, {}))
            loss.backward()

            # clip gradients

            clip_grad_norm_(self.convnet.parameters(), self.max_norm)
            clip_grad_norm_(self.policynet.parameters(), self.max_norm)
            clip_grad_norm_(self.valuenet.parameters(), self.max_norm)

            # push gradients to shared network
            self.share_gradients(self.valuenet, self.shared_value)
            self.share_gradients(self.policynet, self.shared_policy)
            self.share_gradients(self.convnet, self.shared_conv)

            # optimize shared nets
            self.shared_optim.step()
            self.shared_optim.zero_grad()

        # close environment after training
        env.close()

    def action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(dim=0)
            representation = self.convnet(state)
            policy = self.policynet(representation).squeeze(dim=0)
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

