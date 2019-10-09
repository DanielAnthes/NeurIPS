import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, num_in, num_hidden, num_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_in,num_hidden)  # 6*6 from image dimension
        self.fc2 = nn.Linear(num_hidden,num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class REINFORCE_Agent:
    def __init__(self, net):
        self.net        = net
        self.optimizer  = Adam(self.net.parameters(), lr=0.01)
        self.memory     = list()
        self.actions    = actions


    def step(self, state):
        policy  = self.net(state)
        probs   = F.softmax(policy, dim=0).data.numpy()
        probs /= sum(probs) # make sure vector sums to 1
        action  = np.random.choice(actions, size=None, replace=False, p=probs)
        return policy, action


    def learn(self):
        # zero the parameter gradients
        self.optimizer.zero_grad()
        loss = 0
        # calculate loss

        for policies, actions, rewards in self.memory:

            ctg = sum(rewards)
            for i in range(len(rewards)):
                policy = policies[i]
                action = actions[i]

                log_prob = F.log_softmax(policy, dim=0)
                loss += torch.index_select(log_prob, dim=0, index=torch.tensor(action)) * ctg
                ctg -= rewards.pop(0)

        loss *= - (1/len(self.memory)) # adam tries to minimize, so turn around the loss
        loss.backward()
        self.optimizer.step()

        # reset agent fields
        self.memory = list()



print("Initializing Environment...")
env = gym.make("CartPole-v1")
print("Initializing Agent...")
agent = REINFORCE_Agent(Net(4,3,2), [0,1])
print("done.")

n_episodes = 1000
batch_size = 10

total_rewards = np.zeros(n_episodes)
for episode in range(n_episodes):
    state = env.reset()
    done = False
    policies = list()
    actions  = list()
    rewards  = list()
    # run agent and record policies and rewards
    while not done:
        # if episode % 1000 == 0:
        #     env.render()
        state = torch.tensor(state, dtype=torch.float)
        policy, action = agent.step(state)
        state, reward, done, _ = env.step(action)
        policies.append(policy)
        actions.append(action)
        rewards.append(reward)

    agent.memory.append((policies, actions, rewards))
    total_rewards[episode] = sum(rewards)
    if episode % 100 == 0 and episode > 0:
        print("*** EPISODE ", episode, " ***")
        print("mean reward: ", np.mean(total_rewards[episode - 100:episode]))

    # only learn once every batch_size episodes to hopefully reduce variance
    if episode % batch_size == 0 and episode > 0:
        agent.learn()

env.close()

plt.figure()
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
