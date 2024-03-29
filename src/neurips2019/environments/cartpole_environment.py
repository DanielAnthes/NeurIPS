"""Wrapper for the CartPole-v1 Environment"""
import gym
import cv2 

from ..environments.environment import Environment


class CartpoleEnv(Environment):

    def __init__(self, size=(40,40)):
        Environment.__init__(self)
        self.env = gym.make('CartPole-v1')
        self.laststate = None
        self.size = size

    def step(self, action, image=False, diff=False):
        state, reward, done, _ =  self.env.step(action)
        if self.render:
            self.env.render(mode="human")
        if image:
            state = self.env.render(mode="rgb_array") # get state
            state = self.resize(state, self.size) # resize to desired shape
        self.laststate = state
        if diff:
            state = state - self.laststate
        return state, reward, done

    def get_actionspace(self):
        return [0,1]

    def reset(self, image=False):
        state = self.env.reset()
        if self.render:
            self.env.render()
        if image:
            state = self.env.render(mode="rgb_array")
            state = self.resize(state, self.size)
        self.laststate = state
        return state

    def close_window(self):
        self.env.close()

    def resize(self, state, size):
        state = cv2.resize(state, size, interpolation=cv2.INTER_CUBIC) # resize to cnn input size
        state = state.transpose((2, 0, 1)).copy() # convert to pytorch format
        return state
