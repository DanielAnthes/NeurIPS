"""Interface with the environment as given."""
import numpy as np
import socket
from PIL import Image

class Agent:
    def __init__(self):
        pass

    def step(self, end, reward, state):
        # return 0 # nothing
        # return 1 # left
        # return 2 # right
        return   3 # random

class Environment:
    def __init__(self, ip="127.0.0.1", port=13000, size=768, timescale=1, step_cutoff=500, step_reward=0.1, lose_reward=-20, win_factor=2):
        self.client     = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip         = ip
        self.port       = port
        self.size       = size
        self.timescale  = timescale

        max_alive_reward = step_cutoff * step_reward
        self.reward_step = step_reward
        self.reward_lose = lose_reward
        self.reward_win = max(10.0, win_factor * max_alive_reward)

        self.client.connect((ip, port))

    def reset(self):
        self._send(1, 0)
        return self._receive()

    def step(self, action):
        self._send(2, action)
        return self._receive()

    def quit(self):
        self._send(3, 0)

    def state2image(self, state):
        return Image.fromarray(np.array(state, "uint8").reshape(self.size, self.size, 3))

    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data   = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end    = data[0]
        reward = data[1]
        state  = np.array([data[i] for i in range(2, len(data))]).reshape((self.size, self.size, 3)).astype(np.float64)
        state /= 256

        if end:
            if reward == 0:
                reward = self.reward_lose
            else:
                reward = self.reward_win
        else:
            reward += self.reward_step

        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))