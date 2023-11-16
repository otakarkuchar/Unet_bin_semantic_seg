import torch
import torchvision
# from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, utils, transforms
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os


def save_checkpoint(state: dict, model, folder_save_model=".//saved_models//", device="cpu",
                    loss=None, epoch=0, filename="my_model_checkpoint") -> None:
    """
    Save model checkpoint as .pth.tar and .onnx
    :param state:
    :param model: trained model
    :param folder_save_model: folder to save model
    :param device: gpu/cpu
    :param loss: loss function
    :param epoch: n. epoch
    :param filename: name of file
    :return: None
    """

    if not os.path.exists(folder_save_model):
        os.mkdir(folder_save_model)

    print("========= Saving checkpoint =========")

    file_path = folder_save_model + filename
    # torch.save(state, f".//saved_models//model_{loss}_{epoch}.pth.tar")
    torch.save(state, f"{file_path}_{loss}_{epoch}.pth.tar")

    input_names = ["actual_input"]
    output_names = ["output"]
    data_for_onnx = torch.randn(1, 4, 224, 224).to(device=device)
    torch.onnx.export(model,
                      data_for_onnx,
                      f"{file_path}_{loss}_{epoch}.onnx",
                      export_params=True,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=False,
                      opset_version=10,
                      dynamic_axes={'actual_input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


def load_checkpoint(checkpoint, model) -> None:
    """
    Load model checkpoint
    :param checkpoint: loaded checkpoint
    :param model: model to load checkpoint
    :return: None
    """
    print("========= Loading checkpoint =========")
    model.load_state_dict(checkpoint["state_dict"])


def precision_score(target, prediction) -> float:
    intersect = np.sum(prediction * target)
    total_pixel_pred = np.sum(prediction)
    precision = np.mean(intersect / total_pixel_pred)
    return round(precision, 3)


def recall_score(target, prediction) -> float:
    intersect = np.sum(prediction * target)
    total_pixel_truth = np.sum(target)
    recall = np.mean(intersect / total_pixel_truth)
    return round(recall, 3)


def accuracy(targets, predictions) -> float:
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total

    return round(accuracy, 3)


def dice(target, prediction) -> float:
    """
    Compute dice coefficient between target and prediction
    :param target: target mask
    :param prediction: prediction mask from model
    :return: dice coefficient
    """
    smooth = 1.

    flat_predicted = prediction.view(-1)
    flat_gt = target.view(-1)
    intersection = torch.sum(flat_predicted * flat_gt)
    score = (2. * intersection + smooth) / (torch.sum(flat_predicted) + torch.sum(flat_gt) + smooth)

    return score


def check_accuracy(loader, model, device="cuda", epoch=0) -> None:
    """
    Checking accuracy of model
    :param loader: valid / train loader
    :param model: trained model
    :param device: gpu/cpu
    :param epoch: n. epoch
    :return: None
    """
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    with torch.no_grad():
        for idx, data in enumerate(loader):
            x, target = data
            x = x.to(device)
            target = target.to(device)

            x = model(x)
            x = torch.sigmoid(x)
            x = (x > 0.5).float()

            dice_coef = dice(target, x)
            acuracy = accuracy(target, x)

            precision_score_ = precision_score(target.cpu().detach().numpy().flatten(), x.cpu().detach().numpy().flatten())
            recall_score_ = recall_score(target.cpu().detach().numpy().flatten(), x.cpu().detach().numpy().flatten())

            print(f' Metrics for images n.{idx}:'
                  f'epoch = {epoch + 1:d}, precision = {precision_score_:.5f}, '
                  f'recall = {recall_score_:.5f}, 'f'accuracy = {acuracy:.5f}, dice = {dice_coef:.5f}')

            # compute accuracy for all images in batch
            num_correct += (x == target).sum()
            num_pixels += torch.numel(x)

            # compute dice score for all images in batch
            dice_score += (2 * (x * target).sum()) / (
                    (x + target).sum() + 1e-8
            )

        print(f"Got {num_correct}/{num_pixels} with accuracy for all "
              f"valid data is: {num_correct / num_pixels * 100:.2f}")
        print(f"Dice score for all valid data: {dice_score / len(loader)}")

    model.train()


def save_predictions_as_images(loader, model, folder="saved_images/", epoch=0, device="cuda") -> None:
    """
    Save predictions as images
    :param loader: train / valid loader
    :param model: trained model
    :param folder: folder to save images
    :param epoch: n. epoch
    :param device: gpu/cpu
    :return: None
    """
    if not os.path.exists(folder):
        os.mkdir(folder)

    model.eval()
    for idx, (x, target) in enumerate(loader):
        x = x.to(device=device)
        original_image = x
        with torch.no_grad():
            x = model(x)
            x = torch.sigmoid(x)
            x = (x > 0.5).float()

        torchvision.utils.save_image(original_image, f"{folder}/original_image{idx}.png")
        torchvision.utils.save_image(x, f"{folder}/pred_{idx}_epoch{epoch}.png")
        torchvision.utils.save_image(target, f"{folder}{idx}.png")
    model.train()
