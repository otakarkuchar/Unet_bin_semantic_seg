import torch.nn as nn
from dataset import CloudDataset
from model import UNet
from tqdm import tqdm
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_images,
    train_val_loss_as_graph)
from DiceLoss import DiceLoss
import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch.onnx
from kagg_dataloader import CloudDatasetKaggle

# from clearml import Task
# task = Task.init(project_name="my project", task_name="my task")

# Hyperparameters
_LEARNING_RATE: float = 0.00001
_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
_LOSS: str = 'dice'
# _LOSS: str = 'mse'
# _LOSS = 'cross_entropy'
# _LOSS: str = 'bce'
_BATCH_SIZE: int = 32
_NUM_EPOCHS: int = 10000
_NUM_WORKERS: int = 2
_IMAGE_HEIGHT: int = 224
_IMAGE_WIDTH: int = 224
_LOAD_MODEL: bool = True
_NAME_OF_LOAD_MODEL: str = "my_model_checkpoint_dice_10.pth.tar"
_TRAIN_IMG_DIR: str = 'data/train_images/'
_TRAIN_MASK_DIR: str = 'data/train_masks/'
_VALID_IMG_DIR: str = 'data/valid_images/'
_VALID_MASK_DIR: str = 'data/valid_masks/'

# 1) Load Data
train_set_firma = CloudDataset(
    image_dir=_TRAIN_IMG_DIR,
    mask_dir=_TRAIN_MASK_DIR,
    is_trained=True,
    image_height=_IMAGE_HEIGHT,
    image_width=_IMAGE_WIDTH,
    apply_transform=True)

# train_set_kaggle = CloudDatasetKaggle(
#     r_dir=".//data//38-Cloud_training//train_red",
#     g_dir=".//data//38-Cloud_training//train_green",
#     b_dir=".//data//38-Cloud_training//train_blue",
#     nir_dir=".//data//38-Cloud_training//train_nir",
#     gt_dir=".//data//38-Cloud_training//train_gt",
#     pytorch=True)

# base_path = Path('../data/38-Cloud_training')
base_path = Path('C:\\Users\\Tescan_lab\\Documents\\ML segmentation\\Unet_bin_semantic_seg\\data\\38-Cloud_training')
train_set_kaggle = CloudDatasetKaggle(base_path/'train_red',
                    base_path/'train_green',
                    base_path/'train_blue',
                    base_path/'train_nir',
                    base_path/'train_gt')

train_set = torch.utils.data.ConcatDataset([train_set_firma, train_set_kaggle])

valid_set = CloudDataset(
    image_dir=_VALID_IMG_DIR,
    mask_dir=_VALID_MASK_DIR,
    is_trained=False,
    image_height=_IMAGE_HEIGHT,
    image_width=_IMAGE_WIDTH,
    apply_transform=True)


#split the dataset in train and test set
train_set, valid_set = torch.utils.data.random_split(train_set, [8449, 416])
# train_set, valid_set = torch.utils.data.random_split(train_set, [int(len(train_set)*0.99), int(len(train_set)*0.01)])

train_loader = DataLoader(train_set, batch_size=_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valid_set, batch_size=_BATCH_SIZE, shuffle=False)

print(f" Dataset samples: {len(train_set) + len(valid_set)} "
      f"out of which is {len(train_set)} training and "
      f"{len(valid_set)} is validation.")

# 2) Initialize model and send to GPU or load model
model = UNet(in_channels=4, out_channels=1).to(_DEVICE)

if _LOAD_MODEL and os.path.exists(_NAME_OF_LOAD_MODEL):
    load_checkpoint(torch.load(_NAME_OF_LOAD_MODEL), model)
    print("========= Model loaded successfully =========")


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
check_accuracy(loader=val_loader, model=model, criterion=criterion, device=_DEVICE,
               epoch=0, num_epochs=_NUM_EPOCHS)

# 5) Train Network
scaler = torch.cuda.amp.GradScaler()
train_losses = []
val_losses = []
for epoch in range(_NUM_EPOCHS):
    train_loss = 0.0
    loop = tqdm(train_loader)
    for index, data in enumerate(loop):
        input_data, targets = data

        if epoch == 9:
            edit_optimizer(optimizer, lr=0.00001)

        input_data = input_data.to(device=_DEVICE)
        targets = targets.to(device=_DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            output = model(input_data)
            if "data_aug" not in os.listdir():
                os.mkdir("data_aug")
            torchvision.utils.save_image(input_data, f"./data_aug/input{index}_epoch{epoch}.png")
            torchvision.utils.save_image(targets, f"./data_aug/target{index}.png")
            output = torch.sigmoid(output)
            loss = criterion(output, targets)

            train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

    # 6) Check accuracy after training
    avg_val_loss, valid_losses = check_accuracy(loader=val_loader, model=model, criterion=criterion, device=_DEVICE,
                                                epoch=epoch, num_epochs=_NUM_EPOCHS)

    # 7) create graph of train and validation loss
    val_losses.append(avg_val_loss)
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_val_loss_as_graph(avg_train_loss=avg_train_loss, avg_val_loss=avg_val_loss, train_losses=train_losses,
                            val_losses=val_losses, epoch=epoch, num_epochs=_NUM_EPOCHS)

    save_predictions_as_images(
        loader=val_loader, model=model, folder="saved_images/", epoch=epoch, device=_DEVICE)

    if epoch == 9:
        the_best_loss = avg_val_loss

    if avg_val_loss < the_best_loss:
        the_best_loss = avg_val_loss
        # if epoch % 1 == 0:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(state=checkpoint, model=model, device=_DEVICE, loss=_LOSS, epoch=epoch,
                        folder_save_model=".//saved_models//", filename="best_my_model_checkpoint")


    if epoch % 5 == 0:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(state=checkpoint, model=model, device=_DEVICE, loss=_LOSS, epoch=epoch,
                        folder_save_model=".//saved_models//", filename="my_model_checkpoint")
