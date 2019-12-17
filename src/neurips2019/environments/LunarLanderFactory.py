import gym
from neurips2019.environments.EnvironmentFactory import EnvironmentFactory
from neurips2019.environments.LunarLanderEnvironment import LunarLanderEnv
class LunarLanderFactory(EnvironmentFactory):

      def get_instance(self):
            with self.num_instances.get_lock():
                  self.num_instances.value += 1
            return LunarLanderEnv()
