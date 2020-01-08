import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.multiprocessing import Lock, Value, Process, Manager
from neurips2019.agents.agent import Agent
from neurips2019.agents.Worker import Worker
import numpy as np
import matplotlib.pyplot as plt
from neurips2019.util.utils import annealing, slow_annealing
from neurips2019.util.Logger import LogEntry, LogType
import time


# This class implements the Agent interface and the Asynchronous Actor Critic (A3C) algorithm described in "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al)
# This main agent maintains the shared parameters and creates / manages the worker threads
class A3CAgent(Agent):
    """ Main class for an agent that learns and applies the A3C algorithm

    implements the Agent interface and the Asynchronous Actor Critic (A3C) algorithm described in "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al)
    This main agent maintains the shared parameters and creates / manages the worker threads

    Attributes:
        policynet: network used to encode a policy for performing actions in the environment.
        valuenet: network assigning a value to each state in the environment.
        tmax: value determining the number of steps the agent takes before computing the loss during training
        policy_optim: optimizer for the policy network.
        value_optim: optimizer for the value network.
        global_counter: thread safe variable that counts episodes the agent has been trained for.
        env_factory: factory object for generating instances of the environment
        actions: list of allowed actions in the environment.
        config: dictionary holding the configuration for the current instance of the agent
        lock: currently unused, created to allow synchronous updates to the networs by worker processes.
        logq: queue used to process log entries generated in worker processes.
    """

    def __init__(self, config, log_queue):
        """ initialize agent

        reads in the configuration dictionary and initializes attributes

        Args:
            config: a configuration dictionary
            log_queue: queue object for processing of logs generated by worker processes
        """
        # initialize networks
        self.policynet = config["policynet"]()
        self.valuenet = config["valuenet"]()
        self.convnet = config["convnet"]()

        # move models to shared memory
        self.policynet.share_memory()
        self.valuenet.share_memory()
        self.convnet.share_memory()

        self.policynetfunc = config["policynet"] # save 'constructors' of network to create workers
        self.valuenetfunc = config["valuenet"]
        self.convnetfunc = config["convnet"]

        self.tmax = config["lookahead"] # maximum lookahead

        # optimizers
        self.policy_optim = SGD(self.policynet.parameters(), lr=config["policy_lr"], weight_decay=config["policy_decay"])
        self.value_optim = SGD(self.valuenet.parameters(), lr=config["value_lr"], weight_decay=config["value_decay"])
        self.conv_optim = SGD(self.convnet.parameters(), lr=config["conv_lr"], weight_decay=config["conv_decay"])

        self.global_counter = Value('i', 0) # global episode counter
        self.env_factory = config["env"]
        self.actions = config["actions"]
        self.config = config # save config dict
        self.lock = Lock()

        self.logq = log_queue

        # reset iteration counter
        with self.global_counter.get_lock():
            self.global_counter.value = 0


    def train(self, Tmax, num_processes, show_plots=True, render=False):
        """ main training loop for the agent

        initializes worker processes and conducts learning. Each worker process plays episodes in its own instance
        of the environment and updates the shared networks maintained in this agent. This function
        also generates plots with statistics collected during training if this is enabled.

        Args:
            Tmax: number of episodes to be played in total, by all workers combined.
            num_processes: number if worker processes to be spawned
            show_plots: flag to enable / disable plots
            render: if set to True, renders the environment every 100 episodes (if the environment supports being run without rendering)

        Returns:
            return_dict: a dictionary containing information about the training process, logged by the worker processes
        """

        # repeat for training iterations
        manager = Manager()
        return_dict = manager.dict()
        return_dict["scores"] = list()
        processes = list()
        for i in range(num_processes):
            worker = Worker(self.config["entropy"], self.config["entropy_weight"], self.logq, self.policynet, self.valuenet, self.convnet, self.policy_optim, self.value_optim, self.conv_optim, self.global_counter, self.policynetfunc, self.valuenetfunc, self.convnetfunc, self.tmax, self.config["epsilon"], self.env_factory, self.actions, i, self.config["grad_clip"], self.config["gamma"])
            processes.append(Process(target=worker.train, args=(Tmax,return_dict, True, render)))

        # start worker processes
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # after training plot statistics
        if show_plots:
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

        return dict(return_dict)

    def update_networks(self):
        """DEPRECATED updates the policy and value networks maintained by this agent with gradients from worker processes

        uses the optimizers maintained by this agent to update networks using gradients from worker processes.
        Additionally logs the weights of the networks before each update and resets the gradients to zero after updating
        """
        # update networks with gradients from worker processes and reset gradients after
        self.policy_optim.step()
        self.value_optim.step()
        self.policy_optim.zero_grad()
        self.value_optim.zero_grad()

    def action(self, state):
        """
        returns action to be performed according to policy

        Args:
            state: current state of the environment

        Returns:
            policy: policy values for all actions in this state
            action: best action to be performed in this state according to current policy
        """
        state = torch.FloatTensor(state).unsqueeze(dim=0)
        with torch.no_grad():
            representation = self.covnet(state).squeeze(dim=0)
            policy = self.policynet(state)
            probs = F.softmax(policy, dim=0).data.numpy()
            idx = np.argmax(probs)
        return policy, self.actions[idx]

    def calc_loss(self):
        """ dummy loss function required by abstract agent class
        """
        print("loss should be calculated in the Workers")
        return None

    def evaluate(self, num_episodes, show_plots=True, render=True):
        """plays games in the agents environment to evaluate performance

        computes mean score over all evaluation episodes and plots results

        Args:
            num_episodes: number of episodes to play for evaluation
            show_plots:  whether to generate plots for evaluation performance
            render: whether to render the environment while evaluating (if environment supports this)

        Returns:
            scores: scores obtained for each episode
            mean_score: mean score over all played evaluation episodes
        """
        env = self.env_factory.get_instance()
        env.render = render
        scores = list()
        for _ in range(num_episodes):
            episode_reward = 0
            done = False
            state = env.reset(image=True)
            while not done:
                _, action = self.action(state)
                state, reward, done = env.step(action, image=True)
                episode_reward += reward
            scores.append(episode_reward)
        env.close_window()

        self.logq.put(LogEntry(LogType.HISTOGRAM, f"eval/main", np.array(scores), self.global_counter.value, {}))
        mean_score = np.mean(scores)
        print(f"Evaluation mean score: {mean_score}")
        # plot results
        if show_plots:
            plt.figure()
            plt.scatter(range(num_episodes), scores)
            plt.plot([0, num_episodes-1], [mean_score, mean_score], color='orange')
            plt.legend(["mean score", "scores"])

        return scores, mean_score

    def save_model(self, name):
        """ pickles current state of the policynet and valuenet of this agent

        Args:
            name: a filename
        """
        torch.save(self.policynet.state_dict(), f"{name}-policynet.pt")
        torch.save(self.valuenet.state_dict(), f"{name}-valuenet.pt")

    def load_model(self, name):
        """ restores parameters of the policynet and valuenet from a file

        Args:
            name: a filename
        """
        self.policynet.load_state_dict(torch.load(f"{name}-policynet.pt"))
        self.valuenet.load_state_dict(torch.load(f"{name}-valuenet.pt"))
