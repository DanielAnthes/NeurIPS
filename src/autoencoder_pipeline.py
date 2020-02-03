"""Trains an autoencoder on sample data from the NeuroSmash Environment"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import neurips2019.preprocessing.state_autoencoder as state_autoencoder

# hyper parameter
batch_size = 256
learning_rate = 0.005
validation_split = 0.2
log_path = os.path.join("logs", "AutoEncoder")
CUDA = True
epochs = 2


def eval_loop():
    """Runs validation on the autoencoder and writes to tensorboard"""
    print("Entering Eval Loop")
    autoencoder.eval()
    v_losses = 0
    for v_batch, screens in enumerate(valid_dataloader):
        print(f"Validation Step {v_batch}/{len(valid_dataloader)}", end="\r")
        img = screens.view(-1, screensize, screensize, 3).float()
        encoded, decoded = autoencoder(img)
        v_losses += loss_func(decoded, img).cpu().data.numpy()
    print("Writing validation and saving model")
    writer.add_scalar("eval/loss", v_losses / len(valid_dataloader), gstep)
    writer.add_images("eval/decoded", decoded[1:5], gstep)
    writer.add_images("eval/screens", screens[1:5], gstep)
    autoencoder.train()

# move to GPU
if CUDA:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# load data from specified folder
p = os.path.join(os.getcwd(), "logs", "NeuroSmashStates", "run01", "full.npz")
print(f"Loading Dataset from {p}")
screenDataset = state_autoencoder.StateDataset(p)
screensize = screenDataset[0].shape[0]
print(f"Training on dataset of size: {len(screenDataset)}")

# split dataset into train and validation and wrap in a sampler
# https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
indices = list(range(len(screenDataset)))
np.random.shuffle(indices)
split = int(np.floor(validation_split * len(screenDataset)))
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(screenDataset, batch_size=batch_size, sampler=train_sampler)
valid_dataloader = DataLoader(screenDataset, batch_size=batch_size, sampler=valid_sampler)

# initalise tensorboard writer
writer = SummaryWriter(log_path)

# setup network
autoencoder = state_autoencoder.AutoEncoder(screensize=screensize)
if CUDA:
    cuda = torch.cuda.current_device()
    autoencoder = autoencoder.cuda() # .to(cuda, dtype=torch.float32, non_blocking=False)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
loss_func = nn.MSELoss()


eval_time = len(train_dataloader) // 4  # run evaluation four times per epoch
gstep = 0  # globalstep

# run epochs and train autoencoder
for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}")
    losses = []
    for i_batch, screens in enumerate(train_dataloader):
        gstep += 1
        if gstep % 10 == 0:
            print(f"Step {gstep}/{len(train_dataloader)}", end="\r")
        imgs = screens.view(-1, screensize, screensize, 3).float()

        encoded, decoded = autoencoder(imgs)
        loss = loss_func(decoded, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss, gstep)

        # print loss and overview to console if tensorboard is not used
        if writer is None:
            l = loss.data.numpy()
            losses.append(l)
            perdec = i_batch / len(train_dataloader) * 10
            bar = f'{"#" * int(perdec):<9}'
            if len(bar) < 10:
                bar += str(int((perdec * 10)) % 10)
            print(f"[{bar}] || Last Loss: {l:.4f}", end="\r", flush=True)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

        # save a checkpoint every 100 steps
        if gstep % 100 == 0:
            torch.save(autoencoder, os.path.join(log_path, f"checkpoint-{gstep}"))
        # Run evaluation
        if gstep % eval_time == 0:
            eval_loop()

eval_loop()

