from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.agents.Networks import Net
from neurips2019.util.Logger import Logger
from neurips2019.util.utils import annealing, slow_annealing

# Shared hyperparameters
NUM_THREADS = 2

def get_config(env_name:str):
    env = env_name.lower()
    if env in ["lunarlander", "lunar lander", "lunar-lander-v2", "lunarlander-v2"]:
        return get_lunar_lander_config()
    elif env in ["cartpole", "cart pole", "cartpole-v1"]:
        return get_cartpole_config()

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
        "actions": [0,1,2,3] # actions allowed in the environment
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
    cartpole_conf = {
        "valuenet": value_net_cp, # function returning a pytorch network to encode policy
        "policynet": policy_net_cp, # function returning a pytorch network to encode state values
        "train_blocks": 1, # how often train is called
        "block_size": 500, # episodes per call to train
        "num_workers": NUM_THREADS, # number of worker processes
        "lookahead": 30, # steps to take before computing losses
        "show_immediate": False, # show plots after each call to train
        "keep_plots": True, # keep plots open after training has finished
        "debug": False, # additional debug prints
        "epsilon": annealing, # exploration strategy
        "policy_lr": 0.0002, # learning rate for policy net optimizer
        "value_lr": 0.0002, # learning rate for valuenet optimizer
        "policy_decay": 0.0001, # weight decay for policy optimizer
        "value_decay": 0.0001, # weight decay for value optimizer
        "env": CartpoleFactory(), # environment factory object
        "evaluate": 10, # number of episodes to play for evaluation
        "grad_clip": 40, # max norm for gradients, used to clip gradients
        "gamma": 0.99, # discount for future rewards
        "actions": [0,1] # actions allowed in the environment
    }

    return cartpole_conf