from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.agents.A3C import A3CAgent
from neurips2019.agents.Networks import Net
from neurips2019.util.utils import save_agent

def get_policynet():
    return Net(4,10,2)

def get_valuenet():
    return Net(4,10,1)

agent = A3CAgent(10, CartpoleFactory(), [0,1], get_policynet, get_valuenet)
agent.train(100000 ,4) # train for total of 100 episodes, using 4 workers
agent.evaluate(100)
save_agent(agent, "cartpolea3c")
