import torch

def save_agent(agent, name):
    filename = name + ".pt"
    torch.save(agent, filename)


def load_agent(name):
    filename = name + ".pt"
    return torch.load(filename)
