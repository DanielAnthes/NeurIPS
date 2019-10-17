from Agents_debug import ReinforceAgent, DQNAgent, Net, ReplayMemory
from utils import save_agent, load_agent, annealing, random_beginning

resume = False  # So
agent_name = "DQN_cartpole"

if resume:
    agent = load_agent(agent_name)

else:
    agent = DQNAgent(Net(4, 3, 2), Net(4, 3, 2), [0, 1], "CartPole-v1")

agent.train(random_beginning, n_episodes=10000, ctg=False, update_interval=10)

for _ in range(10):
    agent.run()

save_agent(agent, agent_name)
