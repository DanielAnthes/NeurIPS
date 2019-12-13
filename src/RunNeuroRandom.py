from neurosmash_environment import NeurosmashEnvironment
from NeuroRandomAgent import NeuroRandomAgent

env = NeurosmashEnvironment()
agent = NeuroRandomAgent(env)

agent.evaluate()
