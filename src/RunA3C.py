import pickle
import os
import matplotlib.pyplot as plt
from torch.multiprocessing import Queue
from threading import Thread

from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.agents.A3C import A3CAgent
from neurips2019.agents.Networks import Net
from neurips2019.util.utils import annealing, slow_annealing, save_network, load_network
from neurips2019.util.Logger import Logger

from A3C_configs import get_config

# where to save logs
SAVE_DIR = os.path.join("logs","A3C","cartpole_7")

# load saved weights
valuenet_params = "value_weights"
policynet_params = "policy_weights"
convnet_params = "conv_weights"
load_params = False


# initializes agent and runs training loop
def main(config):
    """Trains and evaluates an A3C agent according to config"""
    # set up logger with multiprocessing queue
    print(f"Saving logs to: {SAVE_DIR}")
    queue = Queue()
    logger = Logger(SAVE_DIR, queue)
    # the main instance to run off
    agent = A3CAgent(config, queue)
    if load_params:
        print("Loading parameters...")
        load_network(agent.policynet, policynet_params)
        load_network(agent.valuenet, valuenet_params)
        load_network(agent.convnet, convnet_params)
        print("done.")

    # create logging thread
    log_thread = Thread(target=logger.run, name="logger")
    try:
        # start
        log_thread.start()
        plt.ion() # show plots in a non blocking way

        for i in range(config["train_blocks"]): # train in blocks and save checkpoints
            print(f"Starting Training Block {i}")
            # training process
            result_dict = agent.train(config["block_size"]*(i+1), config["num_workers"], show_plots=False, render=False)
            # evaluation
            agent.evaluate(config["evaluate"], render=False, show_plots=False)

            # save checkpoint
            path = os.path.join(SAVE_DIR, f"checkpoint-{i}")
            agent.save_model(path)

            # show plots of block
            if config["show_immediate"]:
                plt.draw()
                plt.pause(1) # give pyplot time to draw the plots

        # stop and close logger
        queue.put(None)
        log_thread.join()

        # save weights
        print("Saving weights...")
        save_network(agent.policynet, "policy_weights")
        save_network(agent.valuenet, "value_weights")
        save_network(agent.convnet, "conv_weights")
        print("done.")

    except KeyboardInterrupt as e:
        # if interrupt collect thread first
        queue.put(None)
        log_thread.join()
        raise e

    plt.ioff()
    if config["keep_plots"]:
        plt.show() # make sure program does not exit so that plots stay open


# if this file is called as the main entry point for the program, call the main function with parameters specified in config
if __name__ == "__main__":
    main(get_config("cartpole"))
