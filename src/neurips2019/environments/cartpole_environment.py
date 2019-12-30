import gym

from ..environments.environment import Environment


class CartpoleEnv(Environment):

    def __init__(self):
        Environment.__init__(self)
        self.env = gym.make('CartPole-v1')

    def step(self, action):
       state, reward, done, _ =  self.env.step(action)
       if self.render:
           self.env.render()
       return state, reward, done

    def get_actionspace(self):
        return [0,1]

    def reset(self):
        state = self.env.reset()
        if self.render:
            self.env.render()
        return state

    def close_window(self):
        if self.render:
            self.env.close()
