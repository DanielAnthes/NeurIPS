from Logger import LogEntry, LogType
import torch.multiprocessing as mp
import Networks as N
import gym
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from random import random, choice
import numpy as np
from utils import resize, get_state

class Worker(mp.Process):

    def __init__(self, global_counter, global_max_episodes, shared_conv, shared_value, shared_policy, shared_optim, log_queue, name, evaluate, save):
        self.save = save

        # networks
        # self.convnet = N.CNN(128)
        # # self.convnet = N.PretrainedResNet(128)
        # self.valuenet = N.WideNet(128, 32, 1)
        # self.policynet = N.WideNet(128, 32, 2)
        self.convnet = N.GermainNet()
        self.valuenet = N.GermainCritic()
        self.policynet = N.GermainActor(2)

        self.shared_value = shared_value
        self.shared_policy = shared_policy
        self.shared_conv = shared_conv
        self.shared_optim = shared_optim
        self.evaluate = evaluate

        self.global_counter = global_counter
        self.global_max_episodes = global_max_episodes

        self.logq= log_queue

        # parameters
        self.lookahead = 30
        self.gamma = 0.99
        self.actions = [0, 1]
        self.name = name
        self.max_norm = 0.1

    def train(self):
        print("Worker started training")
        reward_eps = list()
        reward_ep = 0
        env = gym.make('CartPole-v1')
        env.reset()
        currentstate = get_state(env)
        state = torch.FloatTensor([currentstate, currentstate, currentstate]).squeeze()
        # state = currentstate

        done = False

        # repeat until maximum number of episodes is reached
        while self.global_counter.value < self.global_max_episodes:
            policy_loss = torch.Tensor([0])
            value_loss = torch.Tensor([0])

            # copy weights from shared net
            self.share_weights(self.shared_policy, self.policynet)
            self.share_weights(self.shared_value, self.valuenet)
            self.share_weights(self.shared_conv, self.convnet)

            states = list()
            actions = list()
            rewards = list()

            for t in range(self.lookahead):
                states.append(state)
                policy, action = self.action(state)
                _ ,reward, done, _ = env.step(action)
                newstate = torch.FloatTensor(get_state(env))
                state = torch.cat((state[1:, :, :], newstate), 0)
                # state = newstate
                actions.append(action)
                rewards.append(reward)
                reward_ep += reward

                if done:
                    env.reset()
                    currentstate = get_state(env)
                    state = torch.FloatTensor([currentstate, currentstate, currentstate]).squeeze()

                    reward_eps.append(reward_ep)
                    # print(f"\n\nEPISODE REWARD {reward_ep}")
                    with self.global_counter.get_lock():
                        self.global_counter.value += 1
                    self.logq.put(LogEntry(LogType.SCALAR, f"reward/{self.name}", reward_ep, self.global_counter.value, {}))
                    reward_ep = 0

                    if self.global_counter.value % 100 == 0:
                        eval_rewards = self.evaluate(10)
                        print(f"MEAN EVALUATION REWARD: {np.mean(eval_rewards)}")
                        for e in eval_rewards:
                            self.logq.put(LogEntry(LogType.SCALAR, f"evaluation", e, self.global_counter.value, {}))
                        self.save("model2")
                    break

            # compute loss over last "lookahead"
            if done:
                R = 0
            else:
                representation = self.convnet(torch.FloatTensor(state).unsqueeze(dim=0))
                R = self.valuenet(representation)

            n_steps = len(rewards)
            policy_loss = 0
            value_loss = 0

            # print("START LOSS COMPUTATION")
            #print(f"number of steps: {n_steps}")
            self.shared_optim.zero_grad() # TODO MAYBE NOT NEEDED
            for t in range(n_steps-1,-1,-1): # traverse backwards through states
                # print(f"step {t}")
                R = rewards[t] + self.gamma * R
                # print(f"reward {R}")
                current_state = torch.FloatTensor(states[t]).unsqueeze(dim=0)
                current_action = torch.LongTensor([actions[t]])
                current_representation = self.convnet(current_state).squeeze(dim=0)
                policy = self.policynet(current_representation)
                # print(f"policy logits {policy}")
                policy = F.log_softmax(policy, dim=0)
                # print(f"log policy {policy}")
                log_policy_t = torch.index_select(policy, dim=0, index=current_action) # policy value of action that was performed
                value_t = self.valuenet(current_representation)
                # print(f"value {value_t}")
                advantage = R - value_t
                # print(f"advantage {advantage}")
                policy_loss -= log_policy_t * advantage.detach()
                # print(f"policy loss {policy_loss}")
                value_loss += advantage**2
                # print(f"value loss {value_loss}")
            loss = 0.1*(0.01*value_loss + policy_loss)

            # normalize loss with lookahead
            # loss /= self.lookahead

            self.logq.put(LogEntry(LogType.SCALAR, f"loss/{self.name}", loss.detach(), self.global_counter.value, {}))
            self.logq.put(LogEntry(LogType.SCALAR, f"value_loss/{self.name}", value_loss.detach(), self.global_counter.value, {}))
            self.logq.put(LogEntry(LogType.SCALAR, f"policy_loss/{self.name}", policy_loss.detach(), self.global_counter.value, {}))

            loss.backward()

            # clip gradients

            clip_grad_norm_(self.convnet.parameters(), self.max_norm)
            clip_grad_norm_(self.policynet.parameters(), self.max_norm)
            clip_grad_norm_(self.valuenet.parameters(), self.max_norm)

            # push gradients to shared network
            self.share_gradients(self.valuenet, self.shared_value)
            self.share_gradients(self.policynet, self.shared_policy)
            self.share_gradients(self.convnet, self.shared_conv)

            if self.name == "Worker-0":
                for idx, (name, param) in enumerate(self.shared_conv.named_parameters()):
                    if "BN" in name: continue
                    self.logq.put(LogEntry(LogType.HISTOGRAM, f"{name}-values", param.flatten().detach(),
                                           self.global_counter.value, {}))
                    self.logq.put(LogEntry(LogType.HISTOGRAM, f"{name}-grads", param.grad.flatten().detach(),
                                           self.global_counter.value, {}))

            # optimize shared nets
            self.shared_optim.step()
            self.shared_optim.zero_grad()

        # close environment after training
        env.close()

    def action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(dim=0)
            representation = self.convnet(state)
            policy = self.policynet(representation).squeeze(dim=0)
            if random() < self.epsilon():
                action = choice(self.actions) # random action
            else:
                probs = F.softmax(policy, dim=0).data.numpy()
                probs /= sum(probs)
                action = np.random.choice(self.actions, size=None, replace=False, p=probs)
        return policy, action


    def epsilon(self):
        eps = max(1 - 2*(self.global_counter.value / self.global_max_episodes), 0.1) # linearly decreasing epsilon as a function of training percentage completed
        if self.name == "Worker-0":
            self.logq.put(LogEntry(LogType.SCALAR, "epsilon", eps, self.global_counter.value, {}))
        return eps

    def share_weights(self, from_net, to_net):
        '''takes two pytorch networks and copies weights from the first to the second network'''
        params = from_net.state_dict()
        to_net.load_state_dict(params)

    def share_gradients(self, from_net, to_net):
        for from_param, to_param in zip(from_net.parameters(), to_net.parameters()):
            to_param._grad = from_param.grad
