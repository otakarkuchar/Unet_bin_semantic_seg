import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import numpy as np
import torchvision.transforms as transforms
import torch
from dataset import CloudDataset
from model import UNet
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)
import time
from dice_loss import dice_loss
import matplotlib.pyplot as plt
import os

# Hyperparameters
LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 30000
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/subscenes/'
TRAIN_MASK_DIR = 'data/cloud_masks/'
# VAL_IMG_DIR = "data/val_images"
# VAL_MASK_DIR = "data/val_masks"



# Load Data
dataset = CloudDataset(
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    transform=transforms.ToTensor(),
)

train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = UNet(in_channels=4, out_channels=3).to(DEVICE)

# Loss and optimizer
# criterion = nn.BCEWithLogitsLoss()
# criterion_entropy = nn.CrossEntropyLoss()
criterion_entropy = nn.MSELoss()
# criterion_entropy = dice_loss

def edit_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if LOAD_MODEL and os.path.exists("my_checkpoint.pth.tar"):
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

check_accuracy(val_loader, model, device=DEVICE, epoch=0)
scaler = torch.cuda.amp.GradScaler()
plot_loss = []

# Train Network
for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader)
    # for batch_idx, (data, targets) in enumerate(train_loader):
    # for batch_idx, (data, targets) in enumerate(loop):
    for index, batch in enumerate(loop):
        data, targets = batch

        # plt.imshow(data[0].permute(1,2,0))
        # plt.show()
        if index == 90:
            edit_optimizer(optimizer, lr=0.0001)

        targets_show = torch.argmax(targets, dim=1).permute(1,2,0)
        show_target = np.zeros((targets_show.shape[0], targets_show.shape[1], 3))
        show_target[:, :, 0] = targets_show[:, :, 0]
        show_target[:, :, 1] = targets_show[:, :, 1]
        show_target[:, :, 2] = targets_show[:, :, 1] * 0

        # plt.imshow(show_target)
        # plt.show()


        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # # forward
        # output = model(data)
        # loss = criterion(output, targets)

        # forward
        with torch.cuda.amp.autocast():
            output = model(data)
            output = torch.softmax(output, dim=1)
            loss = criterion_entropy(output, targets)
            # loss = dice_loss(output, targets)


            # fig, ax = plt.subplots(1, 3, figsize=(15, 8))
            # ax[0].imshow(output[0].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
            # ax[1].imshow(targets[0].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
            # ax[2].imshow(data[0][:3].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
            # plt.show()

            # loss = dice_loss(output, targets)
            # loss = loss1 + loss2
            plot_loss.append(loss.item())

            # if epoch % 100 == 0:
            #     show_output_pytorch = output[0].cpu().permute(1, 2, 0).detach().numpy()
            #     show_output = np.zeros((show_output_pytorch.shape[0], show_output_pytorch.shape[1], 3))
            #     show_output[:, :, 0] = show_output_pytorch[:, :, 0]
            #     show_output[:, :, 1] = show_output_pytorch[:, :, 1]
            #     show_output[:, :, 2] = show_output_pytorch[:, :, 1] * 0
            #
            #     plt.imshow(show_output)
            #     # plt.show()

        targets_show = targets.cpu().numpy()
        mask_show = torch.permute(targets, (1,2,0,3))
        mask_show = mask_show.cpu().detach().numpy()

        # for i in range(8):
        #     plt.imshow(mask_show[:, :, i])
        #     plt.show()


        # # backward
        # optimizer.zero_grad()
        # loss.backward()
        #
        # # gradient descent or adam step
        # optimizer.step()

        optimizer.zero_grad()
        #  Scale Gradients
        scaler.scale(loss).backward()
        #  Update Optimizer
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    # save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if epoch % 1 == 0:
        save_checkpoint(checkpoint, filename="my_checkpoint.pth.tar")

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE, epoch=epoch)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/",epoch=epoch ,device=DEVICE)

        # save figure of loss x = epoch, y = loss
        plt.plot(plot_loss, label='loss')
        plt.savefig(f'loss.png')
        plt.close()


    print(f"Epoch {epoch} Train loss: {loss.item()}")
    # check accuracy
    check_accuracy(val_loader, model, device=DEVICE, epoch=epoch)

