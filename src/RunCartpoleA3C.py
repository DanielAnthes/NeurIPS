from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.agents.A3C import A3CAgent

agent = A3CAgent(10, CartpoleFactory(), [0,1])
agent.train(100,1) # train for total of 100 episodes
