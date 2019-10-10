from Agents import REINFORCE_Agent, Net
from utils import *

resume = True
agent_name = "reinforce_cartpole"

if resume:
    agent = load_agent(agent_name)

else:
    agent = REINFORCE_Agent(Net(4,3,2), [0,1])


agent.train()

for _ in range(10):
    agent.run()

save_agent(agent, agent_name)
