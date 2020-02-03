"""Factory for A3C to produce CartPole envs"""
from neurips2019.environments.EnvironmentFactory import EnvironmentFactory
from neurips2019.environments.cartpole_environment import CartpoleEnv

class CartpoleFactory(EnvironmentFactory):

    def get_instance(self):
        with self.num_instances.get_lock():
            self.num_instances.value += 1
        return CartpoleEnv()
