from environment import Environment
import abc


class EnvironmentFactory:

    def __init__(self):
        self.num_instances = 0


    @abc.abstractmethod
    def get_instance(self):
        '''
        returns a new instance of an environment
        '''
        return
