import abc
from typing import List

from ..environments.environment import Environment

Rewards = List[int]

class Agent(abc.ABC):
    """
    Implements an abstract Agent implementing all necessary methods the agents should share.
    """

    @abc.abstractmethod
    def __init__(self, env:Environment, **kwargs):
        """
        Initialises agent with a given environment which follows the Environment interface.

        Use kwargs to use any additional parameters like layer information your agent needs.
        Use the code below as a guideline.
        """
        self.env = env
        self.actions = env.actionspace
        self.curr_state = env.reset()
        self.memory = None
        self.net = None

    @abc.abstractmethod
    def action(self):
        """
        Takes the next step in the environment from the current state.

        Don't forget to update the current state. Take the code below as inspiration.
        """
        act = self.net.predict(self.curr_state)
        new_state, reward, done = self.env.step(act)
        self.memory.append((self.curr_state, reward, done, new_state))
        self.curr_state = new_state

    @abc.abstractmethod
    def calc_loss(self, **kwargs):
        """
        Calculates the loss. However this is defined in this context.
        """
        pass

    @abc.abstractmethod
    def train(self, **kwargs):
        """
        Trains the network to update the weights.
        """
        pass

    @abc.abstractmethod
    def evaluate(self, runs: int) -> Rewards:
        """
        Without training, plays "runs" rounds in its environment and returns rewards
        """
        pass

    @property
    def environment(self):
        """
        Returns the environment the agent acts upon
        """
        return self.env
