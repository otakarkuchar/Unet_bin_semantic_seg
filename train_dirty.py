import torch.nn as nn
import torchvision.transforms as transforms
import torch
from dataset import CloudDataset
from model import UNet
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)
from DiceLoss import DiceLoss
import matplotlib.pyplot as plt
import os

# Hyperparameters
_LEARNING_RATE = 0.001
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LOSS = 'dice'
_BATCH_SIZE = 16
_NUM_EPOCHS = 30000
_NUM_WORKERS = 2
_IMAGE_HEIGHT = 224
_IMAGE_WIDTH = 224
_LOAD_MODEL = False
_TRAIN_IMG_DIR = 'data/subscenes/'
_TRAIN_MASK_DIR = 'data/cloud_masks/'







# Load Data
dataset = CloudDataset(
    _TRAIN_IMG_DIR,
    _TRAIN_MASK_DIR,
    _IMAGE_HEIGHT,
    _IMAGE_WIDTH,
    transform=transforms.ToTensor(),)

train_set, val_set = torch.utils.data.random_split(dataset, [0.9, 0.1])
train_loader = DataLoader(train_set, batch_size=_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=_BATCH_SIZE, shuffle=False)

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
    raise Exception("Loss not defined")


def edit_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


optimizer = optim.Adam(model.parameters(), lr=_LEARNING_RATE)

if _LOAD_MODEL and os.path.exists("my_checkpoint.pth_dice.tar"):
    load_checkpoint(torch.load("my_checkpoint.pth_dice.tar"), model)

check_accuracy(val_loader, model, device=_DEVICE, epoch=0)
scaler = torch.cuda.amp.GradScaler()
plot_loss = []

# Train Network
for epoch in range(_NUM_EPOCHS):
    loop = tqdm(train_loader)
    # for batch_idx, (input_data, targets) in enumerate(train_loader):
    # for batch_idx, (input_data, targets) in enumerate(loop):
    for index, batch in enumerate(loop):
        data, targets = batch

        # plt.imshow(input_data[0].permute(1,2,0))
        # plt.show()

        if index == 110:
            edit_optimizer(optimizer, lr=0.00001)

        # targets_show = torch.argmax(targets, dim=1).permute(1,2,0)
        # show_target = np.zeros((targets_show.shape[0], targets_show.shape[1], 3))
        # show_target[:, :, 0] = targets_show[:, :, 0]
        # show_target[:, :, 1] = targets_show[:, :, 1]
        # show_target[:, :, 2] = targets_show[:, :, 1] * 0

        # plt.imshow(show_target)
        # plt.show()


        data = data.to(device=_DEVICE)
        targets = targets.to(device=_DEVICE)
        # targets = targets.float().to(device=DEVICE)


        # # forward
        # output = model(input_data)
        # loss = criterion(output, targets)

        # forward
        with torch.cuda.amp.autocast():
            output = model(data)
            output = torch.softmax(output, dim=0)

            loss = criterion(output, targets)


            # for i in range(16):
            #
            #     fig, ax = plt.subplots(1, 3, figsize=(15, 8))
            #     output = torch.concat((output, output[:,0,:,:].unsqueeze(1)*0), dim=1)
            #     ax[0].imshow(output[i].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
            #     targets = torch.concat((targets, targets[:,0,:,:].unsqueeze(1)*0), dim=1)
            #     ax[1].imshow(targets[i].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
            #     ax[2].imshow(input_data[i][:3].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
            #     plt.show()


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
        check_accuracy(val_loader, model, device=_DEVICE, epoch=epoch)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", epoch=epoch, device=_DEVICE)

        # save figure of loss x = epoch, y = loss
        plt.plot(plot_loss, label='loss')
        plt.savefig(f'loss.png')
        plt.close()

    print(f"Epoch {epoch} Train loss: {loss.item()}")