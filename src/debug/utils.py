import time
import subprocess
import cv2
import numpy as np
import torch
from Neurosmash import Environment as NSenv

NEUROSMASH_PATH = r".\Windows\NeuroSmashLite.exe"

def resize(state, size):
        state = cv2.resize(state, size, interpolation=cv2.INTER_CUBIC) # resize to cnn input size
        # state = state.reshape(*size,1)
        state = state.transpose((2, 0, 1)).copy() # convert to pytorch format
        return state

def get_state(env):
    state = env.render(mode="rgb_array")
    # state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = resize(state, (64, 64))
    return torch.FloatTensor(state)

def prep_neurosmash_screen(picture, newmin=0, newmax=255):
    """
    Normalizes an image
    """
    #     dst = np.zeros(picture.shape)
    #     return cv2.normalize(picture, dst=dst, alpha=newmin, beta=newmax, norm_type=cv2.NORM_INF)
    if len(picture.shape) == 2:
        oldmin = picture.min()
        oldmax = picture.max()
        out = (picture - oldmin) * (newmax - newmin) / (oldmax - oldmin) + newmin
    elif len(picture.shape) == 3:
        n, m, d = picture.shape
        s = picture.sum(axis=2).reshape(n, m, 1)
        s[s == 0] = 1
        out = picture.astype(np.float32) / s

    out = out.transpose((2, 0, 1))
    return out

class NeurosmashEnvironment:
    def __init__(self, port, size, timescale):
        self.port = port
        self.size = size
        self.timescale = timescale
        self.env = None

    def __enter__(self):
        subprocess.Popen([
            NEUROSMASH_PATH,
            "-batchmode",
            # "-I", # --ip
            # ip,
            "-P", # --port
            str(self.port),
            "-T", # --timescale
            str(self.timescale),
            "-R", # --resolution
            str(self.size)
        ])
        while self.env is None:
            try:
                self.env = NSenv(port=self.port, size=self.size, timescale=self.timescale)
            except ConnectionRefusedError:
                time.sleep(.1)

        return self.env

    def __exit__(self, *args):
        self.env.quit()
