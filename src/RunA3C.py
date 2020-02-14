"""
Main Entry point for the A3C algorithm. It loads a config file, and sets up the agent and runs the training and eval loop.
"""
import os
import matplotlib.pyplot as plt
from torch.multiprocessing import Queue
from threading import Thread

from neurips2019.agents.A3C import A3CAgent
from neurips2019.util.utils import load_network
from neurips2019.util.Logger import Logger

from A3C_configs import get_config

# where to save logs
SAVE_DIR = os.path.join("logs","A3C","cartpole_res2")

# load saved weights if load_params is true
load_params = False
valuenet_params = os.path.join(SAVE_DIR, "checkpoint-0-valuenet.pt")
policynet_params = os.path.join(SAVE_DIR, "checkpoint-0-policynet.pt")
convnet_params = os.path.join(SAVE_DIR, "checkpoint-0-convnet.pt")


# initializes agent and runs training loop
def main(config):
    """Trains and evaluates an A3C agent according to config"""
    print(f"Saving logs to: {SAVE_DIR}")
    # set up logger with multiprocessing queue
    queue = Queue()
    logger = Logger(SAVE_DIR, queue)
    # the main instance to run off
    agent = A3CAgent(config, queue)
    # load old parameters
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
            print(">> Starting Evaluation")
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
