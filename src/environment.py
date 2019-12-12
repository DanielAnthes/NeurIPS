import abc
from typing import List


class Environment(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, action) -> List[float], int, bool:
        """
        Returns environment as tuple: state, reward, done
        """
        pass

    @abc.abstractmethod
    def get_actionspace(self) -> List:
        """
        Returns available actions as list
        """
        pass

    @abc.abstractmethod
    def reset(self) -> List[float]:
        """
        Returns reset state
        """
        pass
