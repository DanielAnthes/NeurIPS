#%% Imports
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import neurips2019.preprocessing.state_autoencoder as state_autoencoder

batch_size = 8
learning_rate = 0.005
validation_split = 0.12
log_path = os.path.join("logs", "AutoEncoder")
CUDA = True


if CUDA:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

p = os.path.join(os.getcwd(), "logs", "NeuroSmashStates", "run01", "w01", "test.npz")
screenDataset = state_autoencoder.StateDataset(p)#('/neurips2019/preprocessing/states/states_RandomAgent_20191216_193356.npy', screensize=screensize)
screensize = screenDataset[0].shape[0]
print(f"Training on dataset of size: {len(screenDataset)}")

# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
indices = list(range(len(screenDataset)))
np.random.shuffle(indices)
split = int(np.floor(validation_split * len(screenDataset)))
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(screenDataset, batch_size=batch_size, sampler=train_sampler)
valid_dataloader = DataLoader(screenDataset, batch_size=batch_size, sampler=valid_sampler)

writer = SummaryWriter(log_path)

autoencoder = state_autoencoder.AutoEncoder(screensize=screensize)
if CUDA:
    cuda = torch.cuda.current_device()
    autoencoder = autoencoder.cuda() # .to(cuda, dtype=torch.float32, non_blocking=False)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()


epochs = 2

gstep = 0
for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}")
    losses = []
    for i_batch, screens in enumerate(train_dataloader):
        gstep += 1
        imgs = screens.view(-1, screensize, screensize, 3).float()

        encoded, decoded = autoencoder(imgs)
        loss = loss_func(decoded, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss, gstep)

        # l = loss.data.numpy()
        # losses.append(l)
        # perdec = i_batch / len(train_dataloader) * 10
        # bar = f'{"#" * int(perdec):<9}'
        # if len(bar) < 10:
        #     bar += str(int((perdec * 10)) % 10)
        # print(f"[{bar}] || Last Loss: {l:.4f}", end="\r", flush=True)
        # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

        if gstep % 100 == 0:
            autoencoder.eval()
            v_losses = torch.tensor([0])
            for v_batch, screens in enumerate(valid_dataloader):
                img = screens.view(-1, screensize, screensize, 3).float()

                encoded, decoded = autoencoder(img)
                v_losses += loss_func(decoded, img)

            writer.add_scalar("eval/loss", v_losses / len(valid_dataloader), gstep)
            writer.add_images("eval/decoded", decoded[1:5], gstep)
            writer.add_images("eval/screens", screens[1:5], gstep)
            autoencoder.train()

    # print(f"Epoch ended with loss: {losses[-1]:.4f} || Avg Loss: {np.mean(losses):.4f}")


autoencoder.eval()
v_losses = []
for v_batch, screens in enumerate(valid_dataloader):
    img = screens.view(-1, screensize, screensize, 3).float()

    encoded, decoded = autoencoder(img)
    v_losses.append(loss_func(decoded, img).data.numpy())
writer.add_scalar("eval/loss", np.mean(v_losses), gstep)
writer.add_images("eval/decoded", decoded[1:5], gstep)
writer.add_images("eval/screens", screens[1:5], gstep)

#%% Tests
# autoencoder.eval()
# encoded_test, decoded_test = autoencoder(screenDataset[10].float().unsqueeze(0))

