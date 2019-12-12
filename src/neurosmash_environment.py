from environment import Environment
from Neurosmash import Enviroment as NSEnv
class NeurosmashEnvironment(Environment):

    def __init__(self, timescale=1, size=768, port=13000, ip="127.0.0.1"):
        self.env = NSEnv(timescale=timescale)

    def step(self):
        # TODO
        pass

