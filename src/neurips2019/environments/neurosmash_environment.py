from .environment import Environment
from ..util.Neurosmash import Environment as NSEnv


class NeurosmashEnvironment(Environment):

    def __init__(self, timescale=1, size=768, ip="127.0.0.1", port=13000):
        self.env = NSEnv(timescale=timescale, size=size, ip=ip, port=port)

    def step(self, action):
        done, reward, state = self.env.step(action)
        return state, reward, done

    def get_actionspace(self):
        return [0,1,2]

    def reset(self):
        _,_, state = self.env.reset()
        return state

    def close_window(self):
        pass
