import torch
import numpy as np


'''this file is intended to collect utility functions that are reused and not necessarily specific to any particular algorithm'''

def share_weights(from_net, to_net):
    '''takes two pytorch networks and copies weights from the first to the second network'''
    params = from_net.state_dict()
    to_net.load_state_dict(params)


def share_gradients_old(from_net, to_net):
    '''
    copies gradients between networks
    https://discuss.pytorch.org/t/solved-copy-gradient-values/21731
    TODO if asynchronous updating causes problems implement a lock to avoid simultaneous updates to the shared parameters (Note: at the moment this is done by aquiring a lock in the Worker before calling this function)
    '''
    for paramName, paramValue, in from_net.named_parameters():
        for netCopyName, netCopyValue, in to_net.named_parameters():
            if paramName == netCopyName:
                netCopyValue.grad = paramValue.grad.clone()

def share_gradients(from_net, to_net):
    for from_param, to_param in zip(from_net.parameters(), to_net.parameters()):
        to_param._grad = from_param.grad

def save_agent(agent, name):
    # wrapper around torch save function
    filename = name + ".pt"
    torch.save(agent, filename)


def load_agent(name):
    # load saved agents
    filename = name + ".pt"
    return torch.load(filename)

def annealing(episode):
    # specifies the strategy with which the probability of taking a random action changes, returns the current probability of a random action as a function of the number of episodes played
    return (min(1, 0.1 + np.exp(-0.0005 * episode)))

def slow_annealing(episode):
    # specifies the strategy with which the probability of taking a random action changes, returns the current probability of a random action as a function of the number of episodes played
    return (min(1, 0.1 + np.exp(-0.00005 * episode)))

def save_network(network, name):
    torch.save(network.state_dict(), name)

def load_network(network, name):
    network.load_state_dict(torch.load(name))
