from Agents import REINFORCE_Agent, Net
from utils import *
import gym

resume = True
agent_name = "reinforce_lunar_large_discount"

if resume:
    agent = load_agent(agent_name)

else:
    agent = REINFORCE_Agent(Net(8,10,4), [0,1,2,3], "LunarLander-v2")

agent.train(n_episodes=1000)

for _ in range(10):
    agent.run()

save_agent(agent, agent_name)
