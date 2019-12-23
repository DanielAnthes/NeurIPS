import abc
from typing import List

State = List[float]

class Environment(abc.ABC):
    """
    Implements an abstract Environment implementing all necessary methods the environments should share.
    """

    @abc.abstractmethod
    def step(self, action) -> (State, int, bool):
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
    def reset(self) -> State:
        """
        Returns reset state
        """
        pass
