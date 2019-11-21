from DQN import *

# Run the Random Agent
agent = RandomAgent()
environment = Environment()
episode_count = 1000

R0 = np.zeros(episode_count)

for i in range(episode_count):
    reward, state, status = environment.reset()
    print(state)
    while (status[0] == 0):
        action = agent.step(reward, state)
        print(action)
        reward, state, status = environment.step(action)
        R0[i] += reward
