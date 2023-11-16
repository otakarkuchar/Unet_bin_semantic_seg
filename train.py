import torch.nn as nn
import torchvision.transforms as transforms
import torch
from dataset import CloudDataset
from model import UNet
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)
from DiceLoss import DiceLoss
import matplotlib.pyplot as plt
import os
import torch.onnx

# Hyperparameters
_LEARNING_RATE: float = 0.001
_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
_LOSS: str = 'dice'
_LOSS: str = 'mse'
# _LOSS = 'cross_entropy'
# _LOSS: str = 'bce'
_BATCH_SIZE: int = 16
_NUM_EPOCHS: int = 30000
_NUM_WORKERS: int = 2
_IMAGE_HEIGHT: int = 224
_IMAGE_WIDTH: int = 224
_LOAD_MODEL: bool = False
_TRAIN_IMG_DIR: str = 'data/train_images/'
_TRAIN_MASK_DIR: str = 'data/train_masks/'
_VALID_IMG_DIR: str = 'data/valid_images/'
_VALID_MASK_DIR: str = 'data/valid_masks/'


# Load Data
train_set = CloudDataset(
    _TRAIN_IMG_DIR,
    _TRAIN_MASK_DIR,
    True,
    _IMAGE_HEIGHT,
    _IMAGE_WIDTH,
    True)

valid_set = CloudDataset(
    _VALID_IMG_DIR,
    _VALID_MASK_DIR,
    False,
    _IMAGE_HEIGHT,
    _IMAGE_WIDTH,
    True)
    # transform=transforms.ToTensor())

train_loader = DataLoader(train_set, batch_size=_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=_BATCH_SIZE, shuffle=False)

print(f"Number of training examples: {len(train_set)} " + f"Number of validation examples: {len(valid_set)}")

# Model
model = UNet(in_channels=4, out_channels=1).to(_DEVICE)

#  Loss and optimizer
if _LOSS == 'dice':
    criterion = DiceLoss()
elif _LOSS == 'bce':
    criterion = nn.BCEWithLogitsLoss()
elif _LOSS == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()
elif _LOSS == 'mse':
    criterion = nn.MSELoss()
else:
    raise "Loss not defined"


def edit_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


optimizer = optim.Adam(model.parameters(), lr=_LEARNING_RATE)

if _LOAD_MODEL and os.path.exists(f".//saved_models//model_dice_{45}.pth.tar"):
    load_checkpoint(torch.load(f".//saved_models//model_dice_{45}.pth.tar"), model)
    # load_checkpoint(torch.load("my_checkpoint.pth_dice.tar"), model)

check_accuracy(val_loader, model, device=_DEVICE, epoch=0)

scaler = torch.cuda.amp.GradScaler()
train_losses = []
valid_losses = []

# Train Network
for epoch in range(_NUM_EPOCHS):
    loop = tqdm(train_loader)
    for index, batch in enumerate(loop):
        data, targets = batch

        if epoch == 2:
            edit_optimizer(optimizer, lr=0.0001)

        data = data.to(device=_DEVICE)
        targets = targets.to(device=_DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            output = model(data)

            output = torch.sigmoid(output)
            loss = criterion(output, targets)
            train_losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if epoch % 1 == 0:
        save_checkpoint(checkpoint, model=model, device=_DEVICE, loss=_LOSS, epoch=epoch,
                        folder_save_model="./saved_models/", filename="my_model_checkpoint")

        check_accuracy(val_loader, model, device=_DEVICE, epoch=epoch)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", epoch=epoch, device=_DEVICE)

        plt.plot(train_losses, label='loss')
        plt.savefig(f'loss{_LOSS}.png')
        plt.close()

    print(f"Epoch {epoch} Train loss: {loss.item()}")
