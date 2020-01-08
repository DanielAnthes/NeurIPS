#%% Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2
import matplotlib
import os
# matplotlib.use('TkAgg')
matplotlib.interactive(True)

from importlib import reload  # Python 3.4+ only.

import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from neurips2019.preprocessing import neurosmash_state_processing
import neurips2019.preprocessing.state_autoencoder as state_autoencoder

reload(state_autoencoder)


screensize = 128
batch_size = 10
learning_rate = 0.005


p = os.path.join(os.getcwd(), "neurips2019", "preprocessing", "states")
p = os.path.join(p, os.listdir(p)[1])
screenDataset = state_autoencoder.StateDataset(p)#('/neurips2019/preprocessing/states/states_RandomAgent_20191216_193356.npy', screensize=screensize)
state = screenDataset[0]

dataloader = DataLoader(screenDataset, batch_size=batch_size, shuffle=True)

autoencoder = state_autoencoder.AutoEncoder(screensize=screensize)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()

epochs = 50


for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}")
    losses = []
    for i_batch, screens in enumerate(dataloader):
        batch_x = screens.view(-1, screensize, screensize, 3).float()
        batch_y = screens.view(-1, screensize, screensize, 3).float()

        encoded, decoded = autoencoder(batch_x)
        loss = loss_func(decoded, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        perdec = i_batch / len(dataloader) * 10
        bar = f'{"#" * int(perdec):<9}'
        if len(bar) < 10:
            bar += str(int((perdec * 10)) % 10)
        l = loss.data.numpy()
        print(f"[{bar}] || Last Loss: {l:.4f}", end="\r", flush=True)
        losses.append(l)
        # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

    print(f"Epoch ended with loss: {losses[-1]:.4f} || Avg Loss: {np.mean(losses):.4f}")

#%% Tests
autoencoder.eval()
encoded_test, decoded_test = autoencoder(screenDataset[10].float().unsqueeze(0))

plt.imshow(decoded_test.detach().numpy().reshape([screensize, screensize, -1]), cmap="gray")
plt.show()
#%%
# i_screen = 30
# magnitudes = [0.5]*2
#
# screen = np.sum([magnitude * screenDataset[i_screen - i_magnitude] for i_magnitude, magnitude in enumerate(magnitudes)], axis=0)
# screenmod = np.exp(5*screen)
# # screenmod = np.power(screen, 0.1)
# screennorm = (screenmod - np.min(screenmod))/np.max(screenmod)
#
# plt.imshow(screennorm)
# plt.show()
#
#
#
# plt.imshow(screen)
# # plt.show(block=True)
# plt.show()

# cv2.imwrite('screens/screen1.png', screen)
# plt.imshow(screen)
# plt.show()
