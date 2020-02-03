from neurips2019.environments.environment import Environment
import abc
from torch.multiprocessing import Value


class EnvironmentFactory:
    """Implements an interface for env factories used in A3C"""
    def __init__(self):
        self.num_instances = Value('i', 0)

    @abc.abstractmethod
    def get_instance(self) -> Environment:
        '''
        returns a new instance of an environment
        '''
        pass
