import pickle
import os
import matplotlib.pyplot as plt
from torch.multiprocessing import Queue
from threading import Thread
from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.agents.A3C import A3CAgent
from neurips2019.agents.Networks import Net
from neurips2019.util.utils import annealing, slow_annealing
from neurips2019.util.Logger import Logger

from A3C_configs import get_config

SAVE_DIR = "logs"

# initializes agent and runs training loop
def main(config):
    queue = Queue()
    logger = Logger(SAVE_DIR, queue)
    agent = A3CAgent(config, queue)

    log_thread = Thread(target=logger.run, name="logger")
    try:
        log_thread.start()
        plt.ion() # show plots in a non blocking way
        for i in range(config["train_blocks"]): # train in blocks and save checkpoints
            print(f"Starting Training Block {i}")
            result_dict = agent.train(config["block_size"], config["num_workers"], show_plots=False, render=False)
            agent.evaluate(config["evaluate"], render=False, show_plots=False)
            path = os.path.join(SAVE_DIR, f"checkpoint-{i}")
            agent.save_model(path)
            # path = os.path.join(SAVE_DIR, f"results_oneblock_newloss")
            # with open(path, "wb") as f:
            #     pickle.dump(result_dict, f)
            # with open(os.path.join(SAVE_DIR, "weight_log"), "wb") as f:
            #     pickle.dump(agent.weight_log, f)
            if config["show_immediate"]:
                plt.draw()
                plt.pause(1) # give pyplot time to draw the plots
        queue.put(None)
        log_thread.join()
    except KeyboardInterrupt as e:
        queue.put(None)
        log_thread.join()
        raise e

    plt.ioff()
    if config["keep_plots"]:
        plt.show() # make sure program does not exit so that plots stay open
    

# if this file is called as the main entry point for the program, call the main function with parameters specified in config
if __name__ == "__main__":
    main(get_config("cartpole"))
