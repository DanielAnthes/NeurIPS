import torch


'''this file is intended to collect utility functions that are reused and not necessarily specific to any particular algorithm'''

def share_weights(from_net, to_net):
    '''takes two pytorch networks and copies weights from the first to the second network'''
    params = from_net.state_dict()
    to_net.load_state_dict(params)


def share_gradients(from_net, to_net):
    '''copies gradients between networks, inspiration from implementation by 'ikostrikov'
    https://github.com/ikostrikov/pytorch-a3c/blob/48d95844755e2c3e2c7e48bbd1a7141f7212b63f/train.py#L9
    TODO if asynchronous updating causes problems implement a lock to avoid simultaneous updates to the shared parameters
    '''
    from_params = from_net.parameters()
    to_params = to_net.parameters()
    for from_param, to_param in zip(from_params, to_params):
        to_param._grad = from_param.grad

def save_agent(agent, name):
    filename = name + ".pt"
    torch.save(agent, filename)


def load_agent(name):
    filename = name + ".pt"
    return torch.load(filename)

