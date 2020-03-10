import numpy as np
from utils import NeurosmashEnvironment as NSenv

runs = 33
counts = np.zeros(runs)

with NSenv(10002, 64, 1) as e:
    for idx in range(runs):
        i = 0
        e.reset()
        done = False
        while not done:
            done, reward, state = e.step(np.random.randint(0,3))
            i += 1
        counts[idx] = i
