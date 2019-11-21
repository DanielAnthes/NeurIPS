from collections import namedtuple
import numpy as np
import gym

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class PerceptronAgent:
    def __init__(self, environment, state_length, weights=None, actions=[0, 1]):
        if not len(actions) == 2:
            raise NotImplementedError("This perceptron agent can only take two actions")
        self.actions = actions

        self.environment = environment
        self.decide = lambda x: 1 if x >= 0 else 0

        self.state_length = state_length

        if not weights:
            self.weights = np.random.randn(self.state_length)
        else:
            if len(weights) == self.state_length:
                self.weights = weights
            else:
                raise NotImplementedError("Weights should have the same length as the states")

    def _step(self, state):
        state = np.array(state)
        action_idx = self.decide(state @ self.weights)

        return self.actions[action_idx]

    def train(self):
        pass

    def run(self, render=False):
        env = self.environment
        state = env.reset()
        if render:
            env.render()

        done = False
        total_reward = 0

        while not done:
            if render:
                env.render()

            action = self._step(state)
            next_state, reward, done, _ = env.step(action)
            # transition = Transition(state, action, next_state, reward, done)
            total_reward += reward

        env.close()

        return total_reward

    def _average_reward(self, samples, render=False):
        return np.average([self.run(render=render) for _ in range(samples)])

    def _train(self, iterations=10000, samples_per_weight=5, children_per_iteration=100, new_agents_per_iteration=100,
               render=True):
        for iteration in range(iterations):
            average_reward = self._average_reward(samples_per_weight)

            old_weights = self.weights
            new_weights = [self.weights + 1 / 100 * np.random.randn(len(self.weights)) for _ in
                           range(children_per_iteration)]
            new_rewards = [0 for _ in range(children_per_iteration)]

            new_weights += [np.random.randn(len(self.weights)) for i in range(new_agents_per_iteration)]
            new_rewards += [0 for _ in range(new_agents_per_iteration)]

            for i, weights in enumerate(new_weights):
                self.weights = weights
                new_rewards[i] = self._average_reward(samples_per_weight, render=render)

            if np.max(new_rewards) <= average_reward:
                self.weights = old_weights
                print(f"Kept old weights with average reward of {average_reward}")
            else:
                self.weights = new_weights[np.argmax(new_rewards)]
                print(f"Chose new weights to get an average reward of {np.max(new_rewards)}")

        return average_reward


environment_name = "CartPole-v1"
environment_state_len = 4

environment = gym.make(environment_name)
perceptron_agent = PerceptronAgent(environment, environment_state_len, actions=[0, 1])

average_reward = perceptron_agent._train(render=False)

print(average_reward)
