import gym
from EnvironmentFactory import EnvironmentFactory

class CartpoleFactory(EnvironmentFactory):

      def get_instance(self):
            with self.lock:
                  self.num_instances += 1
            return gym.make("CartPole-v1")
