from neurips2019.environments.neurosmash_environment import NeurosmashEnvironment
from neurips2019.agents.NeuroRandomAgent import NeuroRandomAgent

env = NeurosmashEnvironment(size=128)
agent = NeuroRandomAgent(env, save_states=True)

agent.evaluate()
