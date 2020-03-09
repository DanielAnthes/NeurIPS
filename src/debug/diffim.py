import gym
import matplotlib.pyplot as plt
from utils import resize, get_state

env = gym.make("CartPole-v1")
env.reset()
state = get_state(env)
state = state.transpose((1,2,0)).copy() # convert to pytorch format
env.step(1)
newstate = get_state(env)
newstate = newstate.transpose((1,2,0)).copy() # convert to pytorch format
env.close()

plt.figure()
plt.subplot(3,1,1)
plt.imshow(state.squeeze(), cmap="gray")
plt.subplot(3,1,2)
plt.imshow(newstate.squeeze(), cmap="gray")
plt.subplot(3,1,3)
plt.imshow((newstate-state).squeeze(), cmap="gray")
plt.show(block=True)
