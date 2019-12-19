import torch


'''this file is intended to collect utility functions that are reused and not necessarily specific to any particular algorithm'''

def share_weights(from_net, to_net):
    '''takes two pytorch networks and copies weights from the first to the second network'''
    params = from_net.state_dict()
    to_net.load_state_dict(params)


def share_gradients(from_net, to_net):
    '''
    copies gradients between networks
    https://discuss.pytorch.org/t/solved-copy-gradient-values/21731
    TODO if asynchronous updating causes problems implement a lock to avoid simultaneous updates to the shared parameters
    '''
    for paramName, paramValue, in from_net.named_parameters():
        for netCopyName, netCopyValue, in to_net.named_parameters():
            if paramName == netCopyName:
                netCopyValue.grad = paramValue.grad.clone() 


def save_agent(agent, name):
    filename = name + ".pt"
    torch.save(agent, filename)


def load_agent(name):
    filename = name + ".pt"
    return torch.load(filename)

