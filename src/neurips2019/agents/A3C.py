import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.multiprocessing import Lock, Value, Process, Manager
from neurips2019.agents.agent import Agent
from neurips2019.agents.Networks import Net
from neurips2019.agents.Worker import Worker
import numpy as np
import matplotlib.pyplot as plt
from utils import annealing

class A3CAgent(Agent):
    def __init__(self, tmax, env_factory, actions, policynetfunc, valuenetfunc):
        self.policynet = policynetfunc()
        self.valuenet = valuenetfunc()
        self.policynetfunc = policynetfunc # save 'constructors' of network to create workers
        self.valuenetfunc = valuenetfunc
        self.tmax = tmax # maximum lookahead
        self.policy_optim = SGD(self.policynet.parameters(), lr=0.0001)
        self.value_optim = SGD(self.valuenet.parameters(), lr=0.0001)
        self.global_counter = Value('i', 0) # global episode counter
        self.env_factory = env_factory
        self.actions = actions
        self.lock = Lock()

    def train(self, Tmax, num_processes):
        # reset iteration counter
        with self.global_counter.get_lock():
            self.global_counter.value = 0

        # repeat for training iterations
        manager = Manager()
        return_dict = manager.dict()
        return_dict["scores"] = list()
        processes = list()
        for i in range(num_processes):
            worker = Worker(self, self.policynetfunc, self.valuenetfunc, 100, annealing, self.env_factory, self.actions, i)
            processes.append(Process(target=worker.train, args=(Tmax,return_dict)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        plt.figure()
        for i in range(num_processes):
            plt.subplot(num_processes,1,i+1)
            pl = return_dict[f"{i}-policyloss"]
            vl = return_dict[f"{i}-valueloss"]
            plt.plot(range(len(pl)), pl, color="blue")
            plt.plot(range(len(vl)), vl, color="orange")
            plt.legend(["policy loss", "value loss"])
            plt.title(f"worker {i}")

        plt.figure()
        for i in range(num_processes):
            plt.subplot(num_processes,1,i+1)
            scores = return_dict[f"{i}-reward_ep"]
            plt.plot(range(len(scores)), scores, color="orange")
            plt.title(f"worker {i} - scores")

    def update_networks(self):
        self.policy_optim.step()
        self.value_optim.step()

    def action(self, state):
        # performs action according to policy
        # action is picked with probability proportional to policy values
        state = torch.FloatTensor(state)
        policy = self.policynet(state)
        probs = F.softmax(policy, dim=0).data.numpy()
        probs /= sum(probs)  # make sure vector sums to 1
        action = np.random.choice(self.actions, size=None, replace=False, p=probs)
        return policy, action

    def calc_loss(self):
        print("loss should be calculated in the Workers")
        return None

    def evaluate(self, num_episodes):
        env = self.env_factory.get_instance()
        scores = list()
        for _ in range(num_episodes):
            episode_reward = 0
            done = False
            state = env.reset()
            while not done:
                _, action = self.action(state)
                state, reward, done = env.step(action)
                episode_reward += reward
            scores.append(episode_reward)
        # plot results
        plt.figure()
        plt.scatter(range(num_episodes), scores)
        mean_score = np.mean(scores)
        plt.plot([0, num_episodes-1], [mean_score, mean_score], color='orange')
        plt.legend(["mean score", "scores"])
        print(f"mean score: {mean_score}")
        return scores, mean_score


    def save_model(self, name):
        torch.save(self.policynet.state_dict(), f"{name}-policynet.pt")
        torch.save(self.valuenet.state_dict(), f"{name}-valuenet.pt")

    def load_model(self, name):
        self.policynet.load_state_dict(torch.load(f"{name}-policynet.pt"))
        self.valuenet.load_state_dict(torch.load(f"{name}-valuenet.pt"))

