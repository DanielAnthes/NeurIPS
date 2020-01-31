"""
Implements several config dictionaries to be used with A3C.

Use this file to change the configs or create new ones (and if possible link them in `get_config`)
"""

from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.environments.NeurosmashFactory import NeurosmashFactory
from neurips2019.agents.Networks import Net, CNN
from neurips2019.util.Logger import Logger
from neurips2019.util.utils import annealing, slow_annealing

# Shared hyperparameters
NUM_THREADS = 2
STATE_SIZE = (64, 64, 3)

def get_config(env_name:str):
    """Convenience function to get config by string."""
    env = env_name.lower()
    if env in ["lunarlander", "lunar lander", "lunar-lander-v2", "lunarlander-v2"]:
        return get_lunar_lander_config()
    elif env in ["cartpole", "cart pole", "cartpole-v1"]:
        return get_cartpole_config()
    elif env in ["neurosmash", "neuro", "project", "umutsunmut"]:
        return get_neuro_smash()

def get_lunar_lander_config():
    """
    returns lunar lander config dict for A3C
    """
    def policy_net_lunar():
        return Net(8,[32,32],4)
    def value_net_lunar():
        return Net(8, [32,32],1)
    lunar_conf = {
        "valuenet": value_net_lunar, # function returning a pytorch network to encode policy
        "policynet": policy_net_lunar, # function returning a pytorch network to encode state values
        "train_blocks": 1, # how often train is called
        "block_size": 20000, # episodes per call to train
        "num_workers": NUM_THREADS, # number of worker processes
        "lookahead": 30, # steps to take before computing losses
        "show_immediate": False, # show plots after each call to train
        "keep_plots": True, # keep plots open after training has finished
        "debug": False, # additional debug prints
        "epsilon": slow_annealing, # exploration strategy
        "policy_lr": 0.0001, # learning rate for policy net optimizer
        "value_lr": 0.0001, # learning rate for valuenet optimizer
        "policy_decay": 0.0001, # weight decay for policy optimizer
        "value_decay": 0.0001, # weight decay for value optimizer
        "env": LunarLanderFactory(), # environment factory object
        "evaluate": 10, # number of episodes to play for evaluation
        "grad_clip": 40, # max norm for gradients, used to clip gradients
        "gamma": 0.99, # discount for future rewards
        "actions": [0,1,2,3], # actions allowed in the environment
         "entropy": True, # minimize entropy as part of loss function
        "entropy_weight": 10 # weight of entropy in loss function
   }

    return lunar_conf

def get_cartpole_config():
    """
    returns cartpole config dict for A3C
    """
    def policy_net_cp():
        return Net(4,10,2)
    def value_net_cp():
        return Net(4,10,1)
    def conv_net_cp():
        return CNN(STATE_SIZE[0], STATE_SIZE[1], 4)
    cartpole_conf = {
        "valuenet": value_net_cp, # function returning a pytorch network to encode policy
        "policynet": policy_net_cp, # function returning a pytorch network to encode state values
        "convnet": conv_net_cp, # function returning a pytorch network to process image input states
        "train_blocks": 1, # how often train is called
        "block_size": 10, # episodes per call to train
        "num_workers": NUM_THREADS, # number of worker processes
        "lookahead": 30, # steps to take before computing losses
        "show_immediate": False, # show plots after each call to train
        "keep_plots": True, # keep plots open after training has finished
        "debug": False, # additional debug prints
        "epsilon": annealing, # exploration strategy
        "policy_lr": 0.0002, # learning rate for policy net optimizer
        "value_lr": 0.0002, # learning rate for valuenet optimizer
        "conv_lr": 0.0002, # learning rate for convnet
        "policy_decay": 0.0001, # weight decay for policy optimizer
        "value_decay": 0.0001, # weight decay for value optimizer
        "conv_decay": 0.0001, # weight decay for convnet
        "env": CartpoleFactory(), # environment factory object
        "evaluate": 10, # number of episodes to play for evaluation
        "grad_clip": 40, # max norm for gradients, used to clip gradients
        "gamma": 0.99, # discount for future rewards
        "actions": [0,1], # actions allowed in the environment
        "entropy": True, # minimize entropy as part of loss function
        "entropy_weight": 10 # weight of entropy in loss function
    }

    return cartpole_conf

def get_neuro_smash():
    def policy_net():
        return Net(8,[32,16],3)
    def value_net():
        return Net(8,[32,16],1)
    def conv_net():
        return CNN(64, 64, 8)
    conf = {
        "valuenet": value_net, # function returning a pytorch network to encode policy
        "policynet": policy_net, # function returning a pytorch network to encode state values
        "convnet": conv_net, # function returning a pytorch network to process image input states
        "train_blocks": 1, # how often train is called
        "block_size": 10, # episodes per call to train
        "num_workers": NUM_THREADS, # number of worker processes
        "lookahead": 30, # steps to take before computing losses
        "show_immediate": False, # show plots after each call to train
        "keep_plots": True, # keep plots open after training has finished
        "debug": False, # additional debug prints
        "epsilon": annealing, # exploration strategy
        "policy_lr": 0.0002, # learning rate for policy net optimizer
        "value_lr": 0.0002, # learning rate for valuenet optimizer
        "conv_lr": 0.0002, # learning rate for convnet
        "policy_decay": 0.0001, # weight decay for policy optimizer
        "value_decay": 0.0001, # weight decay for value optimizer
        "conv_decay": 0.0001, # weight decay for convnet
        "env": NeurosmashFactory(port=6000, size=64, timescale=5), # environment factory object
        "evaluate": 10, # number of episodes to play for evaluation
        "grad_clip": 40, # max norm for gradients, used to clip gradients
        "gamma": 0.99, # discount for future rewards
        "actions": [0,1], # actions allowed in the environment
        "entropy": True, # minimize entropy as part of loss function
        "entropy_weight": 10 # weight of entropy in loss function
    }

    return conf
