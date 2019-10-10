from Agents import ReinforceAgent, DQNAgent, Net, ReplayMemory
from utils import save_agent, load_agent, annealing

resume = False  # So
agent_name = "DQN_cartpole"

if resume:
    agent = load_agent(agent_name)

else:
    agent = DQNAgent(Net(4, 3, 2), Net(4, 3, 2), [0, 1], "CartPole-v1")

agent.train(annealing, n_episodes=10000)

for _ in range(10):
    agent.run()

save_agent(agent, agent_name)
