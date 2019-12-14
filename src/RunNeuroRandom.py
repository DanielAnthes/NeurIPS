from neurips2019.environments.neurosmash_environment import NeurosmashEnvironment
from neurips2019.agents.NeuroRandomAgent import NeuroRandomAgent

env = NeurosmashEnvironment()
agent = NeuroRandomAgent(env)

agent.evaluate()
