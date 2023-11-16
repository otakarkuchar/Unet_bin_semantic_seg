import torch.nn as nn
from dataset import CloudDataset
from model import UNet
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_images,
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
_NAME_OF_LOAD_MODEL: str = "model_dice_45.pth.tar"
_TRAIN_IMG_DIR: str = 'data/train_images/'
_TRAIN_MASK_DIR: str = 'data/train_masks/'
_VALID_IMG_DIR: str = 'data/valid_images/'
_VALID_MASK_DIR: str = 'data/valid_masks/'


# 1) Load Data
train_set = CloudDataset(
    image_dir=_TRAIN_IMG_DIR,
    mask_dir=_TRAIN_MASK_DIR,
    is_trained=True,
    image_height=_IMAGE_HEIGHT,
    image_width=_IMAGE_WIDTH,
    apply_transform=True)

valid_set = CloudDataset(
    image_dir=_VALID_IMG_DIR,
    mask_dir=_VALID_MASK_DIR,
    is_trained=False,
    image_height=_IMAGE_HEIGHT,
    image_width=_IMAGE_WIDTH,
    apply_transform=True)

train_loader = DataLoader(train_set, batch_size=_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=_BATCH_SIZE, shuffle=False)

print(f" Dataset samples: {len(train_set) + len(valid_set)} "
      f"out of which is {len(train_set)} training and "
      f"{len(valid_set)} is validation.")

# 2) Initialize model and send to GPU or load model
model = UNet(in_channels=4, out_channels=1).to(_DEVICE)

if _LOAD_MODEL and os.path.exists(_NAME_OF_LOAD_MODEL):
    load_checkpoint(torch.load(_NAME_OF_LOAD_MODEL), model)

# 3) Loss and optimizer
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

optimizer = optim.Adam(model.parameters(), lr=_LEARNING_RATE)

def edit_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 4) Check accuracy before training
check_accuracy(val_loader, model, device=_DEVICE, epoch=0)

# 5) Train Network
scaler = torch.cuda.amp.GradScaler()
train_losses = []
valid_losses = []
for epoch in range(_NUM_EPOCHS):
    loop = tqdm(train_loader)
    for index, data in enumerate(loop):
        input_data, targets = data

        if epoch == 2:
            edit_optimizer(optimizer, lr=0.0001)

        input_data = input_data.to(device=_DEVICE)
        targets = targets.to(device=_DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            output = model(input_data)
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
        save_checkpoint(state=checkpoint, model=model, device=_DEVICE, loss=_LOSS, epoch=epoch,
                        folder_save_model=".//saved_models//", filename="my_model_checkpoint")

        check_accuracy(loader=val_loader, model=model, device=_DEVICE, epoch=epoch)

        save_predictions_as_images(
            loader=val_loader, model=model, folder="saved_images/", epoch=epoch, device=_DEVICE)

        plt.plot(train_losses, label='loss')
        plt.savefig(f'loss{_LOSS}.png')
        plt.close()

    print(f"Epoch {epoch} Train loss: {loss.item()}")
