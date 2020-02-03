"""Implements all classes and functions for the autoencoder"""
from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from neurips2019.preprocessing.neurosmash_state_processing import state_to_screen


current_batch = namedtuple("NumpyBatch", ("states", "indices"))


class StateDataset(Dataset):
    """A DataSet class to deal with the neurosmash states"""
    def __init__(self, states_file, screensize=None):
        """
        Args:
            states_file (string): Path to either an npy or an npz file
                                    If npz file, each npy file contained should only contain one single state
        """
        self.states_file = states_file
        states = np.load(states_file)

        if type(states) is np.ndarray:
            self.mode = "array"
            if screensize is None:
                screensize = int(np.sqrt(states.shape[-1] / 3))
            self.screensize = int(screensize)

            screens = []
            for i, s in enumerate(states):
                screens.append(state_to_screen(s, tofloat=True, outsize=self.screensize))

            self.screens = screens

        elif type(states) is np.lib.npyio.NpzFile:
            self.mode = "npz"
            self.states = states
            arr0 = states.files[0]
            if screensize is None:
                screensize = int(np.sqrt(states[arr0].shape[-1] / 3))
            self.screensize = int(screensize)

    def __len__(self):
        if self.mode == "array":
            return len(self.screens)
        elif self.mode == "npz":
            return len(self.states.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode == "array":
            return torch.tensor(self.screens[idx])
        elif self.mode == "npz":
            items = self.states[self.states.files[idx]]
            screens = state_to_screen(items, tofloat=True, outsize=self.screensize)
            return torch.tensor(screens)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, output_shape):
        super(Reshape, self).__init__()
        self.output_shape = output_shape

    def forward(self, input):
        return input.view(-1, *self.output_shape)


class Rollaxis(nn.Module):
    # We need that to bring the channels in order (color, x, y) instead of (x, y, color)
    def forward(self, input):
        if len(input.shape) == 4:
            permutation = (0, 3, 1, 2)
        elif len(input.shape) == 3: # single inference
            permutation = (2, 0, 1)
        return input.permute(*permutation)


class Unrollaxis(nn.Module):
    def forward(self, input):
        if len(input.shape) == 4:
            permutation = (0, 2, 3, 1)
        elif len(input.shape) == 3: # single inference
            permutation = (1, 2, 0)
        return input.permute(*permutation)


class AutoEncoder(nn.Module):
    """Implements an Autoencoder Network"""
    def __init__(self, screensize=128):
        """Init Network for the given screensize of states"""
        super(AutoEncoder, self).__init__()

        def calc_dim(dim, kernel_size, stride=1, padding=1, dilation=1):
            """Conveience function to calculate dimensionality after convolution"""
            out = int(((dim + 2*padding - dilation*(kernel_size - 1)) // stride) + 0.5)
            return out

        out_dim = screensize
        self.encoder1 = nn.Sequential()
        self.encoder1.add_module("rollaxis", Rollaxis())
        self.encoder1.add_module("init_batchnorm", nn.BatchNorm2d(3))
        self.encoder1.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, dilation=1))
        out_dim = calc_dim(out_dim, 3, 1, 1, 1)
        self.encoder1.add_module("act1", nn.SELU(True))
        self.encoder1.add_module("conv2", nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1))
        out_dim = calc_dim(out_dim, 3, 2, 1, 1)
        self.encoder1.add_module("act2", nn.SELU(True))

        self.encoder_pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        out_dim = out_dim // 2

        self.encoder2 = nn.Sequential()
        self.encoder2.add_module("conv3", nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1))
        out_dim = calc_dim(out_dim, 3, 2, 1, 1)
        self.encoder2.add_module("flatten", Flatten())
        self.encoder2.add_module("ffn_batchnorm", nn.BatchNorm1d(out_dim**2 * 16))
        self.encoder2.add_module("ffn", nn.Linear(out_dim**2 * 16, 1024))
        self.encoder2.add_module("readout", nn.Linear(1024, 128))

        self.decoder1 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.Linear(1024, out_dim**2 * 16),
            Reshape((16, out_dim, out_dim)),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.decoder_unpool = nn.MaxUnpool2d(2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.SELU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SELU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            Unrollaxis()
        )

    def forward(self, x):
        # print("Input", x.shape)
        encoded = self.encoder1(x)
        encoded, indices = self.encoder_pool(encoded)
        encoded = self.encoder2(encoded)
        # print("Encod", encoded.shape)
        # print("Indic", indices.shape)
        decoded = encoded
        decoded = self.decoder1(decoded)
        # print("UConv", decoded.shape)
        decoded = self.decoder_unpool(decoded, indices)
        decoded = self.decoder2(decoded)
        # print("Decod", decoded.shape)

        # for name, mod in self.decoder._modules.items():
        #     print(decoded.shape, "-->", name)
        #     decoded = mod(decoded)
        # decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(epochs=10, batch_size=64, learning_rate=0.005):
    pass
