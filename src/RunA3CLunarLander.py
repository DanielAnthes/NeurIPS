from neurips2019.environments.CartpoleFactory import CartpoleFactory
from neurips2019.agents.A3C import A3CAgent
from neurips2019.agents.Networks import Net
import matplotlib.pyplot as plt

def get_policynet():
    return Net(4,25,2)

def get_valuenet():
    return Net(4,25,1)


### CONFIG
num_train_blocks = 1
block_size = 10000
num_workers = 16
lookahead = 30

show_immediate = False # show plots after each training set
keep_plots = True # show plots after script has finished
debug = True
###

agent = A3CAgent(lookahead, CartpoleFactory(), [0,1], get_policynet, get_valuenet)

plt.ion() # show plots in a non blocking way
for i in range(num_train_blocks): # train in blocks and save checkpoints
    print(f"Starting Training Block {i}")
    agent.train(block_size ,num_workers) # train for total of 10000 episodes, using 4 workers
    agent.evaluate(500)
    agent.save_model(f"checkpoint-{i}")
    if show_immediate:
        plt.draw()
        plt.pause(1) # give pyplot time to draw the plots

    if debug:
        print("########### DEBUG ############")
        print("policynet weights")
        print(agent.policynet.state_dict())
        print("########### \DEBUG ############\n\n")


plt.ioff()
if keep_plots:
    plt.show() # make sure program does not exit so that plots stay open
