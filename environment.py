import abc

class Environment(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass

    @abc.abstractmethod
    def get_actionspace(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass
