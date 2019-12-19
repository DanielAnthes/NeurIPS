from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.agents.A3C import A3CAgent
from neurips2019.agents.Networks import Net
from neurips2019.agents.utils import save_agent

def get_policynet():
    return Net(8,25,4)

def get_valuenet():
    return Net(8,25,1)

agent = A3CAgent(50, LunarLanderFactory(), [0,1,2,3], get_policynet, get_valuenet)
agent.train(100000 ,8) # train for total of 10000 episodes, using 4 workers
agent.evaluate(100)
save_agent(agent, "LunarA3Cfinal")
