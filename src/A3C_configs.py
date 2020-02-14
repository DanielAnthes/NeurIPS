"""
Implements several config dictionaries to be used with A3C.

Use this file to change the configs or create new ones (and if possible link them in `get_config`)
"""

from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.environments.NeurosmashFactory import NeurosmashFactory
from neurips2019.agents.Networks import Net, CNN, WideNet, PretrainedResNet
from neurips2019.util.utils import annealing, slow_annealing, linear_annealing
import torch.nn as nn



# Shared hyperparameters
NUM_THREADS = 4


def get_config(env_name:str):
    """Convenience function to get config by string."""
    env = env_name.lower()
    if env in ["cartpole", "cart pole", "cartpole-v1"]:
        return get_cartpole_config()
    elif env in ["neurosmash", "neuro", "project"]:
        return get_neuro_smash()


def get_cartpole_config():
    """
    returns cartpole config dict for A3C
    """
    conv_out = 64
    def policy_net_cp():
        return Net(conv_out, 2)
    def value_net_cp():
        return Net(conv_out, 1)
    def conv_net_cp():
        return CNN(conv_out)
    def resnet_cp():
        return PretrainedResNet(conv_out)


    cartpole_conf = {
        "valuenet": value_net_cp, # function returning a pytorch network to encode policy
        "policynet": policy_net_cp, # function returning a pytorch network to encode state values
        "convnet": resnet_cp, # function returning a pytorch network to process image input states
        "train_blocks": 1, # how often train is called
        "block_size": 1000, # episodes per call to train
        "num_workers": NUM_THREADS, # number of worker processes
        "lookahead": 10, # steps to take before computing losses
        "show_immediate": False, # show plots after each call to train
        "keep_plots": True, # keep plots open after training has finished
        "debug": False, # additional debug prints
        "epsilon": annealing, # exploration strategy
        "policy_lr": 0.01, # learning rate for policy net optimizer
        "value_lr": 0.0002, # learning rate for valuenet optimizer
        "conv_lr": 0.05, # learning rate for convnet
        "policy_decay": 0.01, # weight decay for policy optimizer
        "value_decay": 0.0001, # weight decay for value optimizer
        "conv_decay": 0.0001, # weight decay for convnet
        "env": CartpoleFactory(), # environment factory object
        "evaluate": 50, # number of episodes to play for evaluation
        "grad_clip": 10000000000, # max norm for gradients, used to clip gradients
        "gamma": 0.99, # discount for future rewards
        "actions": [0,1], # actions allowed in the environment
        "entropy": False, # minimize entropy as part of loss function
        "entropy_weight": -0.1, # weight of entropy in loss function
        "frameskip": 0
    }

    return cartpole_conf


def get_neuro_smash():
    """ returns the config dict for the neurosmash env"""
    conv_out = 128
    def policy_net():
        return WideNet(conv_out, 32, 3)
    def value_net():
        return WideNet(conv_out, 32, 1)
    def conv_net():
        return CNN(conv_out)
    conf = {
        "valuenet": value_net, # function returning a pytorch network to encode policy
        "policynet": policy_net, # function returning a pytorch network to encode state values
        "convnet": conv_net, # function returning a pytorch network to process image input states
        "train_blocks": 1, # how often train is called
        "block_size": 2500, # episodes per call to train
        "num_workers": NUM_THREADS, # number of worker processes
        "lookahead": 100, # steps to take before computing losses
        "show_immediate": False, # show plots after each call to train
        "keep_plots": True, # keep plots open after training has finished
        "debug": False, # additional debug prints
        "epsilon": annealing, # exploration strategy
        "policy_lr": 0.0001, # learning rate for policy net optimizer
        "value_lr": 0.0001, # learning rate for valuenet optimizer
        "conv_lr": 0.0005, # learning rate for convnet
        "policy_decay": 0.0001, # weight decay for policy optimizer
        "value_decay": 0.0001, # weight decay for value optimizer
        "conv_decay": 0.0001, # weight decay for convnet
        "env": NeurosmashFactory(port=8000, size=64, timescale=5), # environment factory object
        "evaluate": 30, # number of episodes to play for evaluation
        "grad_clip": 1, # max norm for gradients, used to clip gradients
        "gamma": 0.99, # discount for future rewards
        "actions": [0,1,2], # actions allowed in the environment
        "entropy": True, # minimize entropy as part of loss function
        "entropy_weight":-1, # weight of entropy in loss function
        "frameskip": 0 # repeat action for x frames
    }

    return conf
