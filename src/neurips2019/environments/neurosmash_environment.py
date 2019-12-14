from .environment import Environment
from ..util.Neurosmash import Environment as NSEnv


class NeurosmashEnvironment(Environment):

    def __init__(self, timescale=1, size=768, port=13000, ip="127.0.0.1"):
        self.env = NSEnv(timescale=timescale)

    def step(self, action):
        done, reward, state = self.env.step(action)
        return state, reward, done

    def get_actionspace(self):
        return [0,1,2]

    def reset(self):
        _,_, state = self.env.reset()
        return state
