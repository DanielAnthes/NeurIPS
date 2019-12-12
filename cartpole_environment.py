from environment import Environment
import gym

class CartpoleEnv(Environment):

    def __init__(self):
        self.env = gym.make('CartPole.v1')

    def step(self, action):
       state, reward, done, _ =  self.env.step(action)
       return state, reward, done

    def get_actionspace(self):
        return [0,1]

    def reset(self):
        state = self.env.reset()
        return state
