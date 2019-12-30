import matplotlib.pyplot as plt
from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.agents.A3C import A3CAgent
from neurips2019.agents.Networks import Net
from neurips2019.util.utils import annealing, slow_annealing

# initializes agent and runs training loop
def main(config):
    agent = A3CAgent(config)

    plt.ion() # show plots in a non blocking way
    for i in range(config["train_blocks"]): # train in blocks and save checkpoints
        print(f"Starting Training Block {i}")
        agent.train(config["block_size"] ,config["num_workers"]) # train for total of 10000 episodes, using 4 workers
        agent.evaluate(config["evaluate"])
        agent.save_model(f"checkpoint-{i}")
        if config["show_immediate"]:
            plt.draw()
            plt.pause(1) # give pyplot time to draw the plots

        if config["debug"]: # your debug statements here
            print("########### DEBUG ############")
            print("########### \DEBUG ############\n\n")

    plt.ioff()
    if config["keep_plots"]:
        plt.show() # make sure program does not exit so that plots stay open


### CONFIG FOR LUNAR ###
lunar_conf = {"valuenet": lambda: Net(8, [32,32],1), # function returning a pytorch network to encode policy
              "policynet": lambda: Net(8,[32,32],4), # function returning a pytorch network to encode state values
              "train_blocks": 1, # how often train is called
              "block_size": 10000, # episodes per call to train
              "num_workers": 8, # number of worker processes
              "lookahead": 30, # steps to take before computing losses
              "show_immediate": False, # show plots after each call to train
              "keep_plots": True, # keep plots open after training has finished
              "debug": False, # additional debug prints
              "epsilon": annealing, # exploration strategy
              "policy_lr": 0.0001, # learning rate for policy net optimizer
              "value_lr": 0.0001, # learning rate for valuenet optimizer
              "policy_decay": 0.01, # weight decay for policy optimizer
              "value_decay": 0.01, # weight decay for value optimizer
              "env": LunarLanderFactory(), # environment factory object
              "evaluate": 500, # number of episodes to play for evaluation
              "actions": [0,1,2,3]} # actions allowed in the environment
###


# if this file is called as the main entry point for the program, call the main function with parameters specified below
if __name__ == "__main__":
    main(lunar_conf)
