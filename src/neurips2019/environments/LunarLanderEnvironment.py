import gym

from ..environments.environment import Environment


class LunarLanderEnv(Environment):

    def __init__(self):
        self.env = gym.make('LunarLander-v2')

    def step(self, action):
       state, reward, done, _ =  self.env.step(action)
       return state, reward, done

    def get_actionspace(self):
        return [0,1,2,3]

    def reset(self):
        state = self.env.reset()
        return state
