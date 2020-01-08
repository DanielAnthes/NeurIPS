import gym

from ..environments.environment import Environment


class CartpoleEnv(Environment):

    def __init__(self):
        Environment.__init__(self)
        self.env = gym.make('CartPole-v1')

    def step(self, action, image=False):
        state, reward, done, _ =  self.env.step(action)
        if self.render:
            self.env.render(mode="human")
        if image:
            state = self.env.render(mode="rgb_array").transpose((2, 0, 1)).copy()
        return state, reward, done

    def get_actionspace(self):
        return [0,1]

    def reset(self, image=False):
        state = self.env.reset()
        if self.render:
            self.env.render()
        if image:
            state = self.env.render(mode="rgb_array").transpose((2,0,1)).copy()
        return state

    def close_window(self):
        if self.render:
            self.env.close()
