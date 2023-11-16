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


def save_checkpoint(state: dict, model, folder_save_model="./saved_models/", device="cpu",
                    loss=None, epoch=0, filename="my_model_checkpoint") -> None:

    if not os.path.exists(folder_save_model):
        os.mkdir(folder_save_model)

    print("=> Saving checkpoint")

    file_path = folder_save_model + filename
    torch.save(state, f".//saved_models//model_{loss}_{epoch}.pth.tar")
    # torch.save(state, f" {file_path}_{loss}_{epoch}.pth.tar")

    input_names = ["actual_input"]
    output_names = ["output"]
    data_for_onnx = torch.randn(1, 4, 224, 224).to(device=device)
    torch.onnx.export(model,
                      data_for_onnx,
                      f".//saved_models//_{loss}_{epoch}.onnx",
                      export_params=True,
                      input_names=input_names,
                      output_names=output_names,
                      verbose=False,
                      opset_version=10,
                      dynamic_axes={'actual_input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})


def load_checkpoint(checkpoint, model) -> None:
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def plot_confusion_matrix(m, title='Confusion matrix') -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    img = ax.matshow(m)
    plt.title(title)
    plt.colorbar(img)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"], rotation=45)
    plt.yticks(tick_marks, ["0", "1"])
    plt.tight_layout()
    plt.ylabel("A")
    plt.xlabel("B")
    # plt.show()


def precision_score_(ground_truth_mask, pred_mask) -> float:
    intersect = np.sum(pred_mask * ground_truth_mask)
    total_pixel_pred = np.sum(pred_mask)
    precision = np.mean(intersect / total_pixel_pred)
    return round(precision, 3)


def recall_score_(ground_truth_mask, pred_mask) -> float:
    intersect = np.sum(pred_mask * ground_truth_mask)
    total_pixel_truth = np.sum(ground_truth_mask)
    recall = np.mean(intersect / total_pixel_truth)
    return round(recall, 3)


def accuracy(groundtruth_mask, pred_mask) -> float:
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3)


def generalized_dice_coefficient(y_true, y_pred) -> float:
    smooth = 1.
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
            np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score


def dice_coef(ground_truth_mask, pred_mask) -> float:
    intersect = np.sum(pred_mask * ground_truth_mask)
    total_sum = np.sum(pred_mask) + np.sum(ground_truth_mask)
    dice = np.mean(2*intersect/total_sum)
    return round(dice, 3)  #round up to 3 decimal places


def dice_loss(y_true, y_pred) -> float:
    loss = 1 - generalized_dice_coefficient(y_true, y_pred)
    return loss


def check_accuracy(loader, model, device="cuda", epoch=0):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # x = model(x)
            # x = torch.softmax(x, dim=0)
            x = torch.sigmoid(model(x))

            x = (x > 0.5).float()

            precision_score_1 = precision_score_(y.cpu().detach().numpy().flatten(), x.cpu().detach().numpy().flatten())
            recall_score_1 = recall_score_(y.cpu().detach().numpy().flatten(), x.cpu().detach().numpy().flatten())
            accuracy_1 = accuracy(y.cpu().detach().numpy().flatten(), x.cpu().detach().numpy().flatten())
            dice_1 = dice_coef(y.cpu().detach().numpy().flatten(), x.cpu().detach().numpy().flatten())

            dice_loss_value = dice_loss(y.cpu().detach().numpy().flatten(), x.cpu().detach().numpy().flatten())

            print(f'epoch = {epoch + 1:d}, precision = {precision_score_1:.5f}, recall = {recall_score_1:.5f},'
                  f'accuracy = {accuracy_1:.5f}, dice = {dice_1:.5f}', f' dice_loss = {dice_loss_value:.5f}')

            # if True:
            #     for j in range(len(y)):
            #     # label = y[j].cpu().detach().numpy().flatten()
            #     # prediction = x[j].cpu().detach().numpy().flatten()
            #
            #         label = y[j].permute(1, 2, 0).cpu().detach().numpy()*125
            #         label = transfer_to_3D_numpy(label)
            #         prediction = x[j].permute(1, 2, 0).cpu().detach().numpy()*125
            #         prediction = transfer_to_3D_numpy(prediction)
            #
            #         fig, ax = plt.subplots(1, 2, figsize=(15, 8))
            #         ax[0].imshow(label)
            #         ax[1].imshow(prediction)
            #         plt.show()

            x = torch.argmax(x, dim=1)
            y = torch.argmax(y, dim=1)

                # confusion_m = confusion_matrix(label, prediction)
                # plot_confusion_matrix(confusion_m)

    model.train()


def save_predictions_as_imgs(
        loader, model, folder="saved_images/", epoch=0, device="cuda") -> None:
    if not os.path.exists(folder):
        os.mkdir(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        orig = x
        with torch.no_grad():

            # x = model(x)
            # x = torch.softmax(x, dim=0)
            x = torch.sigmoid(model(x))
            x = (x > 0.5).float()
            # x = x.squeeze(1)
            # y = y.to(torch.long)

            # x = torch.argmax(x, dim=1)
            # y = torch.argmax(y, dim=1)
            # x_show = x.unsqueeze(1).float()
            # y_show = y.unsqueeze(1).float()
            # x_show = x[idx].cpu().detach().numpy()
            # y_show = y[idx].cpu().detach().numpy()

            # v pripade 2D?
            # x = torch.concat((x, x[:,0,:,:].unsqueeze(1)*0), dim=1)
            # y = torch.concat((y, y[:,0,:,:].unsqueeze(1)*0), dim=1)

            # Add an additional channel
            # new_channel_data = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]))
            #
            # # Concatenate the new channel along the channel dimension
            # x_show = torch.cat((x, new_channel_data), dim=1)
        # xx = torchvision.utils.make_grid(x)
        torchvision.utils.save_image(orig, f"{folder}/orig{idx}.png")
        torchvision.utils.save_image(x, f"{folder}/pred_{idx}_epoch{epoch}.png")
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()
