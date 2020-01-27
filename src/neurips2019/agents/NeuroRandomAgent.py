from random import choice
import numpy as np
from neurips2019.preprocessing import neurosmash_state_processing

from neurips2019.agents.agent import Agent


class NeuroRandomAgent(Agent):
    """
    A random agent acting as baseline
    """
    
    def __init__(self, env, save_states=False, save_dir=None):
        self.env = env
        self.actions = env.get_actionspace()
        self.save_states = save_states
        self.states = []
        self.save_dir = save_dir

    def action(self, *args, **kwargs):
        return choice(self.actions)

    def calc_loss(self):
        # This is ugly, but since this agent wont ever learn anything we do not need to compute an actual loss
        return 0

    def train(self):
        # random agents cannot learn
        pass

    def evaluate(self, **kwargs):
        save_start_state = kwargs.get("save.start_state", None)

        state = self.env.reset()
        rewards = list()
        done = False
        steps = 0

        while not done:
            steps += 1
            state, reward, done = self.env.step(self.action())
            if self.save_states:
                self.states.append(state)
            # if done:
            #     print("Done")
            # else:
            #     print(f'Step {steps}', end="\r")
            rewards.append(reward)

        if self.save_states:
            if save_start_state is not None:
                self.states = self.states[save_start_state:]
            # neurosmash_state_processing.save_states(self.states, "RandomAgent", self.save_dir)

        return np.sum(rewards), rewards
