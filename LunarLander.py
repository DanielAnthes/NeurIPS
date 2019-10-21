from Agents import ReinforceAgent, DQNAgent, Net
from utils import *
import gym

resume = False
agent_name = "DQN_lunar"

if resume:
    agent = load_agent(agent_name)

else:
    agent = DQNAgent(Net(8, 10, 4), Net(8, 10, 4), [0, 1, 2, 3], "LunarLander-v2")

agent.train(random_beginning, n_episodes=10000, ctg=True)

for _ in range(10):
    agent.run()

save_agent(agent, agent_name)
