import torch
import numpy as np

def save_agent(agent, name):
    filename = name + ".pt"
    torch.save(agent, filename)


def load_agent(name):
    filename = name + ".pt"
    return torch.load(filename)

def annealing(episode):
    return(min(1, 0.1 + np.exp(-0.0005 * episode)))
