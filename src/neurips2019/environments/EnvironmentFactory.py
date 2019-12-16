from environment import Environment
import abc
from multithreading import Lock


class EnvironmentFactory:

    def __init__(self):
        self.num_instances = 0
        self.lock = Lock()


    @abc.abstractmethod
    def get_instance(self):
        '''
        returns a new instance of an environment
        '''
        return
