import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2
import matplotlib.pyplot as plt
import numpy as np
from neurips2019.preprocessing import neurosmash_state_processing


class StateDataset(Dataset):

    def __init__(self, states_file, screensize):
        """
        Args:
            states_file (string): Path to the list of states
        """
        self.states_file = states_file
        self.screensize = screensize

        states = np.load(states_file)
        screens = [neurosmash_state_processing.state_to_screen(state, tofloat=True, outsize=self.screensize) for state in states]
        # self.empty_screen = np.mean(screens, axis=0)
        self.screens = [torch.tensor(screen) for screen in screens]

        # for state in states:
        #     self.screens.append(torch.tensor(neurosmash_state_processing.state_to_screen(state, outsize=self.screensize)))

    def __len__(self):
        return len(self.screens)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.screens[idx]

    def show(self, idx):
        plt.imshow(self[idx])
        plt.show()

    def show_empty(self):
        plt.imshow(self.empty_screen)
        plt.show()


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
        permutation = (0, 3, 1, 2)
        return input.permute(*permutation)


class Unrollaxis(nn.Module):
    def forward(self, input):
        permutation = (0, 2, 3, 1)
        return input.permute(*permutation)

class AutoEncoder(nn.Module):

    def __init__(self, screensize=128):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Conv2d(screensize, 256, 3),
            Rollaxis(),
            nn.Conv2d(3, 32, kernel_size=2, stride=1, padding=1),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=2), # reduce to half size
            nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(True),
            # nn.MaxPool2d(2, stride=1),  # reduce to quarter size


            # nn.Linear(screensize * screensize * 3, 1024),
            # nn.Tanh(),
            # nn.Linear(1024, 512),
            # nn.Tanh(),
            # nn.Linear(512, 256),
            # nn.Tanh(),
            # nn.Linear(256, 100),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 2, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 2, stride=1, padding=1),  # b, 8, 15, 15
            # nn.ReLU(True),
            # nn.ConvTranspose2d(8, 3, 2, stride=2, padding=0),  # b, 1, 28, 28
            nn.Tanh(),
            Unrollaxis(),

            # nn.Conv2d(3, 16, kernel_size=2, stride=2, padding=1),
            # nn.ReLu(True),
            # nn.MaxPool2d(2, stride=2),  # reduce to half size
            # nn.MaxPool2d(2, stride=1),  #
            # nn.ReLU(True),
            # nn.Conv2d(16, 3, kernel_size=2, stride=2, padding=1),
            #
            # Unrollaxis(),
            # nn.Linear(100, 256),
            # nn.Tanh(),
            # nn.Linear(256, 512),
            # nn.Tanh(),
            # nn.Linear(512, 1024),
            # nn.Tanh(),
            # nn.Linear(1024, (screensize * screensize * 3)),
            # # nn.Conv2d(256, screensize, 3),
            # nn.Sigmoid(),  # compress to a range (0, 1)
            # Reshape([screensize, screensize, 3]),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(epochs=10, batch_size=64, learning_rate=0.005):
    pass
