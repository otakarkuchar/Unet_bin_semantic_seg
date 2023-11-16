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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        intersection = torch.sum(y_true * y_pred)
        score = (2. * intersection + self.smooth) / (
                torch.sum(y_true) +
                torch.sum(y_pred) + self.smooth)

        return 1. - score




# def generalized_dice_coefficient(y_true, y_pred):
#     smooth = 1.
#     # y_true_f = y_true.cpu().detach().numpy().flatten()
#     # y_pred_f = y_pred.cpu().detach().numpy().flatten()
#     intersection = torch.sum(y_true * y_pred)
#     score = (2. * intersection + smooth) / (
#             torch.sum(y_true) + torch.sum(y_pred) + smooth)
#     return score
#
# def dice_loss(y_true=None, y_pred=None):
#     loss = 1 - generalized_dice_coefficient(y_true, y_pred)
#     return loss
