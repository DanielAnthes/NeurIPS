import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.multiprocessing import Lock, Value, Process, Manager
from neurips2019.agents.agent import Agent
from neurips2019.agents.Worker import Worker
import numpy as np
import matplotlib.pyplot as plt
from neurips2019.util.utils import annealing, slow_annealing


# This class implements the Agent interface and the Asynchronous Actor Critic (A3C) algorithm described in "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al)
# This main agent maintains the shared parameters and creates / manages the worker threads
class A3CAgent(Agent):
    def __init__(self, config):
        # initialize networks
        self.policynet = config["policynet"]()
        self.valuenet = config["valuenet"]()
        self.policynetfunc = config["policynet"] # save 'constructors' of network to create workers
        self.valuenetfunc = config["valuenet"]
        self.tmax = config["lookahead"] # maximum lookahead

        # optimizers
        self.policy_optim = SGD(self.policynet.parameters(), lr=config["policy_lr"], weight_decay=config["policy_decay"])
        self.value_optim = SGD(self.valuenet.parameters(), lr=config["value_lr"], weight_decay=config["value_decay"])

        self.global_counter = Value('i', 0) # global episode counter
        self.env_factory = config["env"]
        self.actions = config["actions"]
        self.lock = Lock()

    def train(self, Tmax, num_processes):
        # main train loop, spawns worker threads
        # reset iteration counter
        with self.global_counter.get_lock():
            self.global_counter.value = 0

        # repeat for training iterations
        manager = Manager()
        return_dict = manager.dict()
        return_dict["scores"] = list()
        processes = list()
        for i in range(num_processes):
            worker = Worker(self, self.policynetfunc, self.valuenetfunc, self.tmax, annealing, self.env_factory, self.actions, i)
            processes.append(Process(target=worker.train, args=(Tmax,return_dict)))

        # start worker processes
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # after training plot statistics
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
        # update networks with gradients from worker processes and reset gradients after
        self.policy_optim.step()
        self.value_optim.step()
        self.policy_optim.zero_grad()
        self.value_optim.zero_grad()

    def action(self, state):
        # performs action according to policy
        # performs action that has highest policy value for the given state
        state = torch.FloatTensor(state)
        policy = self.policynet(state)
        probs = F.softmax(policy, dim=0).data.numpy()
        idx = np.argmax(probs)
        return policy, self.actions[idx]

    def calc_loss(self):
        print("loss should be calculated in the Workers")
        return None

    def evaluate(self, num_episodes):
        # play games in the agents environment, number of games to be played is passed as a parameter
        # computes mean score and plots results
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

