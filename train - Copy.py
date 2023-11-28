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


class Trainer():
    def __init__(self, img_dir=_TRAIN_IMG_DIR, mask_dir=_TRAIN_MASK_DIR, image_height=_IMAGE_HEIGHT,
                 image_width=_IMAGE_WIDTH, load_model=_LOAD_MODEL,batch_size=_BATCH_SIZE,
                 num_epochs=_NUM_EPOCHS, loss=_LOSS, num_workers=_NUM_WORKERS,
                 learning_rate=_LEARNING_RATE, device=_DEVICE, transform=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_height = image_height
        self.image_width = image_width
        self.load_model = load_model
        self.batch_size = batch_size
        self.loss = loss
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.device = device
        self.transform = transform



        # Load Data
        self.dataset = CloudDataset(
            self.img_dir,
            self.mask_dir,
            self.image_height,
            self.image_width,
            transform=transforms.ToTensor(),
        )

        train_set, val_set = torch.utils.data.random_split(self.dataset, [0.9, 0.1])
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        # Model
        self.model = UNet(in_channels=4, out_channels=1).to(self.device)

        #  Loss and optimizer
        if self.loss == 'dice':
            self.criterion = DiceLoss()
        elif self.loss == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise Exception("Loss not defined")

        def edit_optimizer(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.load_model and os.path.exists("my_checkpoint.pth_dice.tar"):
            load_checkpoint(torch.load("my_checkpoint.pth_dice.tar"), self.model)

        check_accuracy(self.val_loader, self.model, device=self.device, epoch=0)
        self.scaler = torch.cuda.amp.GradScaler()
        self.plot_loss = []

    # Train Network
    def train(self):
        for epoch in range(self.num_epochs):
            loop = tqdm(self.train_loader)
            # for batch_idx, (input_data, targets) in enumerate(train_loader):
            # for batch_idx, (input_data, targets) in enumerate(loop):
            for index, batch in enumerate(loop):
                data, targets = batch

                # plt.imshow(input_data[0].permute(1,2,0))
                # plt.show()

                if index == 110:
                    edit_optimizer(self.optimizer, lr=0.00001)

                # targets_show = torch.argmax(targets, dim=1).permute(1,2,0)
                # show_target = np.zeros((targets_show.shape[0], targets_show.shape[1], 3))
                # show_target[:, :, 0] = targets_show[:, :, 0]
                # show_target[:, :, 1] = targets_show[:, :, 1]
                # show_target[:, :, 2] = targets_show[:, :, 1] * 0

                # plt.imshow(show_target)
                # plt.show()


                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                # targets = targets.float().to(device=DEVICE)


                # forward
                with torch.cuda.amp.autocast():
                    output = self.model(data)

                    if self.loss != 'cross_entropy':
                        output = torch.softmax(output, dim=0)

                    loss = self.criterion(output, targets)
                    self.plot_loss.append(loss.item())

                    # for i in range(16):
                    #
                    #     fig, ax = plt.subplots(1, 3, figsize=(15, 8))
                    #     output = torch.concat((output, output[:,0,:,:].unsqueeze(1)*0), dim=1)
                    #     ax[0].imshow(output[i].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
                    #     targets = torch.concat((targets, targets[:,0,:,:].unsqueeze(1)*0), dim=1)
                    #     ax[1].imshow(targets[i].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
                    #     ax[2].imshow(input_data[i][:3].cpu().permute(1, 2, 0).detach().numpy().astype(np.float32))
                    #     plt.show()



                    # if epoch % 100 == 0:
                    #     show_output_pytorch = output[0].cpu().permute(1, 2, 0).detach().numpy()
                    #     show_output = np.zeros((show_output_pytorch.shape[0], show_output_pytorch.shape[1], 3))
                    #     show_output[:, :, 0] = show_output_pytorch[:, :, 0]
                    #     show_output[:, :, 1] = show_output_pytorch[:, :, 1]
                    #     show_output[:, :, 2] = show_output_pytorch[:, :, 1] * 0
                    #
                    #     plt.imshow(show_output)
                    #     # plt.show()

                self.optimizer.zero_grad()
                #  Scale Gradients
                self.scaler.scale(loss).backward()
                #  Update Optimizer
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update tqdm loop
                loop.set_postfix(loss=loss.item())

            # save model
            checkpoint = {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            if epoch % 1 == 0:
                save_checkpoint(checkpoint, filename="my_checkpoint.pth.tar")

                # check accuracy
                check_accuracy(self.val_loader, self.model, device=self.device, epoch=epoch)

                # print some examples to a folder
                save_predictions_as_imgs(
                    self.val_loader, self.model, folder="saved_images/", epoch=epoch, device=self.device)

                # save figure of loss x = epoch, y = loss
                plt.plot(self.plot_loss, label='loss')
                plt.savefig(f"loss.png")
                plt.close()

            print(f"Epoch {epoch} Train loss: {loss.item()}")


if __name__ == "__main__":
    train = Trainer()
    train.train()
