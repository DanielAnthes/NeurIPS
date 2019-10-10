from Agents import REINFORCE_Agent, DQN_Agent, Net
from utils import *

resume     = False
agent_name = "DQN_cartpole"

if resume:
    agent = load_agent(agent_name)

else:
    agent = DQN_Agent(Net(4,3,2), Net(4,3,2), [0,1], "CartPole-v1")


agent.train(n_episodes=300)

for _ in range(10):
    agent.run()

save_agent(agent, agent_name)
