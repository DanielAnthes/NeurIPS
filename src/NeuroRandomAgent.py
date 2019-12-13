from agent import Agent
from random import choice
import numpy as np

class NeuroRandomAgent(Agent):

    def __init__(self, env):
        self.env = env
        self.actions = env.get_actionspace()

    def action(self):
        return choice(self.actions)

    def calc_loss(self):
        # This is ugly, but since this agent wont ever learn anything we do not need to compute an actual loss
        return 0

    def train(self):
        # random agents cannot learn
        pass

    def evaluate(self):
        state = self.env.reset()
        rewards = list()
        done = False

        while not done:
            done, reward, _ = self.env.step(self.action())
            rewards.append(reward)

        return np.sum(rewards), rewards
