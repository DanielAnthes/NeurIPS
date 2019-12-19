from neurips2019.environments.LunarLanderFactory import LunarLanderFactory
from neurips2019.agents.A3C import A3CAgent
from neurips2019.agents.Networks import Net

def get_policynet():
    return Net(8,25,4)

def get_valuenet():
    return Net(8,25,1)


### CONFIG
num_train_blocks = 4
block_size = 1000
num_workers = 8
###

agent = A3CAgent(50, LunarLanderFactory(), [0,1,2,3], get_policynet, get_valuenet)
for i in range(num_train_blocks): # train in blocks and save checkpoints
    agent.train(block_size ,num_workers) # train for total of 10000 episodes, using 4 workers
    agent.evaluate(100)
    agent.save_model(f"checkpoint-{i}")
