import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):

        prediction = prediction.contiguous()
        target = target.contiguous()

        intersection = (prediction * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + self.smooth) / (
                    prediction.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        score = loss.mean()

        return score


