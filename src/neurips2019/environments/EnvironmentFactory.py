from neurips2019.environments.environment import Environment
import abc
from torch.multiprocessing import Value


class EnvironmentFactory:

    def __init__(self):
        self.num_instances = Value('i', 0)

    @abc.abstractmethod
    def get_instance(self) -> Environment:
        '''
        returns a new instance of an environment
        '''
        pass
