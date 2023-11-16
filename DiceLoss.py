import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):

        flat_predicted = y_pred.view(-1)
        flat_gt = y_true.view(-1)
        intersection = torch.sum(flat_predicted * flat_gt)

        return 1 - ((2. * intersection + self.smooth) /
                    (torch.sum(flat_predicted) + torch.sum(flat_gt) + self.smooth))

