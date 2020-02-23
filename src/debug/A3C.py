from utils import resize, get_state
import numpy as np
import gym
import torch
from random import random, choice
import Networks as N
import itertools
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.multiprocessing import Value, Process
from Worker import Worker

class A3C:

    def __init__(self, queue):
        # networks
        self.convnet = N.CNN(128)
        self.valuenet = N.WideNet(128, 32, 1)
        self.policynet = N.WideNet(128, 32, 2)

        self.convnet.share_memory()
        self.valuenet.share_memory()
        self.policynet.share_memory()

        params = [self.convnet.parameters(), self.valuenet.parameters(), self.policynet.parameters()]
        # self.optimizer = Adam(itertools.chain(*params))
        self.optimizer = SGD(itertools.chain(*params), lr=0.0001, momentum=0.9)

        self.global_counter = Value('i', 0)

        self.log_queue = queue

        self.actions = [0, 1]

    def train(self, num_processes, episodes):
        # set up a list of processes
        processes = list()
        for i in range(num_processes):
            worker = Worker(self.global_counter, episodes, self.convnet, self.valuenet, self.policynet, self.optimizer, self.log_queue, f"Worker-{i}", self.evaluate)
            processes.append(Process(target=worker.train))

        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def evaluate(self, num_eps):
        env = gym.make("CartPole-v1")
        done = False
        rewards = list()
        for i in range(num_eps):
            ep_reward = 0
            done = False
            env.reset()
            state = get_state(env)
            while not done:
                _, action = self.action(torch.FloatTensor(state))
                state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = get_state(env)
            print(f"REWARD: {ep_reward}")
            rewards.append(ep_reward)
        env.close()
        return rewards

    def action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(dim=0)
            representation = self.convnet(state)
            policy = self.policynet(representation).squeeze(dim=0)
            probs = F.softmax(policy, dim=0).data.numpy()
            probs /= sum(probs)
            action = np.random.choice(self.actions, size=None, replace=False, p=probs)
        return policy, action

