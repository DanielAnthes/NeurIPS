import numpy as np

from neurips2019.environments.environment import Environment
from neurips2019.util.Neurosmash import Environment as NSEnv
from neurips2019.preprocessing.neurosmash_state_processing import state_to_screen as s2s


class NeurosmashEnvironment(Environment):

    def __init__(self, timescale=1, size=768, ip="127.0.0.1", port=13000):
        self.env = NSEnv(timescale=timescale, size=size, ip=ip, port=port)
        self.size = size

    def step(self, action, **kwargs):
        done, reward, state = self.env.step(action)
        return self.__to_screen(state), reward, done

    def get_actionspace(self, **kwargs):
        return [0,1,2]

    def reset(self, **kwargs):
        _,_, state = self.env.reset()
        return self.__to_screen(state)

    def close_window(self, **kwargs):
        pass

    def __to_screen(self, state):
        # out = np.array(state, "uint8").reshape(self.size, self.size, 3)
        out = s2s(state, size=self.size, outsize=40, tofloat=True, norm=True)
        return out.transpose((2, 0, 1))