import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    # Adapted from https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self, num_in, num_hidden, num_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_in, num_hidden)  # 6*6 from image dimension
        self.fc2 = nn.Linear(num_hidden, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNAgent:
    def __init__(self, qnet: Net, qhatnet: Net, actions, env_name, replay_memory=ReplayMemory(capacity=500), gamma=1, epsilon=0.9,
                 minibatch_size=100, learning_rate=0.1):
        self.qnet = qnet
        self.optimizer = Adam(self.qnet.parameters(), lr=learning_rate)
        self.qhatnet = qhatnet
        self.actions = actions
        self.env_name = env_name
        self.gamma = gamma
        self.epsilon = epsilon
        self.minibatch_size = minibatch_size

        self.replay_memory = replay_memory

    def _step(self, state):
        # "toss a coin" to decide whether to take a random action
        coin = np.random.rand()

        # if random action
        if coin <= self.epsilon:
            return random.choice(self.actions)

        else:
            state = torch.tensor(state, dtype=torch.float)
            Qvals = self.qnet(state).detach().numpy()

            # maxQ = np.max(Qvals)
            # action = np.where(Qvals == maxQ)[0][0]

            # We can save complexity by directly taking the index of the best action
            action = np.argmax(Qvals)
            return action

    def _loss(self, transition: Transition):
        # state, action, reward, new_state, done = transition
        state, action, next_state, reward, done = transition
        state = torch.tensor(state, dtype=torch.float)

        # compute y
        if done:
            y = reward
        else:
            next_state = torch.tensor(next_state, dtype=torch.float)
            Q_vals = self.qhatnet(next_state)
            # Q_action = torch.index_select(Q_vals, dim=0, index=torch.tensor(action))
            Q_action = Q_vals.max()
            y = reward + self.gamma * Q_action

        Qs = self.qnet(state)
        Q = torch.index_select(Qs, dim=0, index=torch.tensor(action))
        return (Q - y) ** 2

    def train(self, epsilon_func, n_episodes=3000, update_interval=100):
        env = gym.make(self.env_name)
        total_rewards = np.zeros(n_episodes)

        for episode in range(n_episodes):
            state = env.reset()
            self.epsilon = epsilon_func(episode)
            done = False
            while not done:
                action = self._step(state)
                next_state, reward, done, _ = env.step(action)
                new_state = next_state
                total_rewards[episode] += reward
                transition = Transition(state, action, next_state, reward, done)
                self.replay_memory.push(transition)

            # only learn once buffer is filled
            if len(self.replay_memory) >= self.replay_memory.capacity:

                # sample minibatch
                batch = self.replay_memory.sample(self.minibatch_size)

                loss = 0.0
                self.optimizer.zero_grad()
                for transition in batch:
                    loss += self._loss(transition)

                loss.backward()
                self.optimizer.step()

                if episode > 0 and episode % update_interval == 0:
                    self.qhatnet.load_state_dict(self.qnet.state_dict())

            if episode % 100 == 0 and episode > 0:
                print("*** EPISODE ", episode, " ***")
                print("mean reward: ", np.mean(total_rewards[episode - 100:episode]))
                print(f"Epsilon: {self.epsilon}")

    def run(self):
        env = gym.make(self.env_name)
        state = env.reset()
        env.render()
        done = False
        total_reward = 0

        while not done:
            env.render()
            state = torch.tensor(state, dtype=torch.float)
            action = self._step(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        env.close()
        print("Total Reward: ", total_reward)
        return total_reward


class ReinforceAgent:
    def __init__(self, net, actions, env_name, learning_rate=0.01, discount=0.9):
        self.net = net
        self.optimizer = Adam(self.net.parameters(), lr=learning_rate)
        self.memory = list()
        self.actions = actions
        self.env_name = env_name
        self.discount = discount

    def _step(self, state):
        policy = self.net(state)
        probs = F.softmax(policy, dim=0).data.numpy()
        probs /= sum(probs)  # make sure vector sums to 1
        action = np.random.choice(self.actions, size=None, replace=False, p=probs)
        return policy, action

    def _learn(self):
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
                ctg *= self.discount

        loss *= - (1 / len(self.memory))  # adam tries to minimize, so turn around the loss
        loss.backward()
        self.optimizer.step()

        # reset agent fields
        self.memory = list()

    def train(self, n_episodes=1000, batch_size=10):
        env = gym.make(self.env_name)

        total_rewards = np.zeros(n_episodes)
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            policies = list()
            actions = list()
            rewards = list()
            # run agent and record policies and rewards
            while not done:
                # if episode % 1000 == 0:
                #     env.render()
                state = torch.tensor(state, dtype=torch.float)
                policy, action = self._step(state)
                state, reward, done, _ = env.step(action)
                policies.append(policy)
                actions.append(action)
                rewards.append(reward)

            self.memory.append((policies, actions, rewards))
            total_rewards[episode] = sum(rewards)
            if episode % 100 == 0 and episode > 0:
                print(f"*** EPISODE {episode} ***")
                print(f"mean reward: {np.mean(total_rewards[episode - 100:episode])}")

            # only learn once every batch_size episodes to hopefully reduce variance
            if episode % batch_size == 0 and episode > 0:
                self._learn()
        env.close()
        plt.figure()
        plt.plot(total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.show()

    def run(self):
        env = gym.make(self.env_name)
        state = env.reset()
        env.render()
        done = False
        total_reward = 0

        while not done:
            env.render()
            state = torch.tensor(state, dtype=torch.float)
            policy, action = self._step(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        env.close()
        print("Total Reward: ", total_reward)
        return total_reward
