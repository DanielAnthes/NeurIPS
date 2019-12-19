from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.agents.A3C import A3CAgent
from neurips2019.agents.Networks import Net
import matplotlib.pyplot as plt

def get_policynet():
    return Net(8,25,4)

def get_valuenet():
    return Net(8,25,1)


### CONFIG
num_train_blocks = 4
block_size = 300
num_workers = 8
show_immediate = False # show plots after each training set
keep_plots = False # show plots after script has finished
###

agent = A3CAgent(50, LunarLanderFactory(), [0,1,2,3], get_policynet, get_valuenet)

plt.ion() # show plots in a non blocking way
for i in range(num_train_blocks): # train in blocks and save checkpoints
    print(f"Starting Training Block {i}")
    agent.train(block_size ,num_workers) # train for total of 10000 episodes, using 4 workers
    agent.evaluate(100)
    agent.save_model(f"checkpoint-{i}")
    if show_immediate:
        plt.draw()
        plt.pause(1) # give pyplot time to draw the plots
plt.ioff()
if keep_plots:
    plt.show() # make sure program does not exit so that plots stay open
