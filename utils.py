import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple


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
    # dice = []

    # for i in range(len(target)):
    #     fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    #     ax[0].imshow(target[i].cpu().permute(1, 2, 0).detach().numpy())
    #     ax[1].imshow(prediction[i].cpu().permute(1, 2, 0).detach().numpy())
    #     plt.show()
    #
    #     flat_predicted = prediction[i].view(-1)
    #     flat_gt = target[i].view(-1)
    #     intersection = torch.sum(flat_predicted * flat_gt)
    #     score = (2. * intersection + smooth) / (torch.sum(flat_predicted) + torch.sum(flat_gt) + smooth)
    #     # print(f'Dice score for image n.{i} is {score:.5f}')
    #     dice.append(score)
    #
    # print(f'Mean dice score for batch is {torch.mean(torch.stack(dice)):.5f}')
    # # prediction = prediction.squeeze(1)
    # # target = target.squeeze(1)
    # flat_predicted = prediction.view(-1)
    # flat_gt = target.view(-1)
    # intersection = torch.sum(flat_predicted * flat_gt)
    # score = (2. * intersection + smooth) / (torch.sum(flat_predicted) + torch.sum(flat_gt) + smooth)
    # print(f'Dice score for image is {score:.5f}')
    #
    # smooth = 1.
    #
    # # have to use contiguous since they may from a torch.view op
    # iflat = prediction.contiguous().view(-1)
    # tflat = target.contiguous().view(-1)
    # intersection = (iflat * tflat).sum()
    #
    # # A_sum = torch.sum(iflat * iflat)
    # # B_sum = torch.sum(tflat * tflat)
    # A_sum = torch.sum(tflat)
    # B_sum = torch.sum(iflat)
    #
    # dice_git = ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    # print(f'Dice score for image is {dice_git:.5f}')
    #
    #
    # numerator = 2 * torch.sum(prediction * target)
    # denominator = torch.sum(prediction + target)
    # dice_net = (numerator + 1) / (denominator + 1)
    # print(f'Dice score for image is {dice_net:.5f}')


    import torchmetrics
    # dic = torchmetrics.functional.f1(target, prediction, num_classes=2, average='macro', mdmc_average='global')
    # from torchmetrics import F1Score
    # f1 = F1Score(num_classes=2, task='binary')
    # f1(prediction, target)


    pred = prediction.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    score = loss.mean()

    return score

def jacard_similarity(target, prediction) -> float:
    """
     Intersection-Over-Union (IoU), also known as the Jaccard Index
    :param target: target mask
    :param prediction: prediction mask from model
    :return: jacard similarity
    """
    flat_predicted = prediction.view(-1)
    flat_gt = target.view(-1)

    intersection = torch.sum(flat_gt * flat_predicted)
    union = torch.sum((flat_gt + flat_predicted) - (flat_gt * flat_predicted))
    return intersection / union


def check_accuracy(loader, model, criterion, device="cuda", epoch=0, num_epochs=0) -> Tuple[float, list]:
    """
    Checking accuracy of model
    :param loader: valid / train loader
    :param model: trained model
    :param device: gpu/cpu
    :param epoch: epoch
    :param num_epochs: n. epochs to train
    :param criterion: loss function
    :return: None
    """
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    val_loss = 0.0
    valid_losses = []
    jacaaard_similarity_dataset = 0
    dice_score_dataset = 0
    accuracy_dataset = 0
    with torch.no_grad():
        for idx, data in enumerate(loader):
            x, target = data
            x = x.to(device)
            target = target.to(device)

            x = model(x)
            x = torch.sigmoid(x)
            loss = criterion(x, target)
            val_loss += loss.item()
            valid_losses.append(loss.item())

            x = (x > 0.5).float()

            dice_coef = dice(target, x)
            dice_score_dataset += dice_coef
            acuracy = accuracy(target, x)
            accuracy_dataset += acuracy
            jacaard_similarity = jacard_similarity(target, x)
            jacaaard_similarity_dataset += jacaard_similarity

            print(f' Metrics for images batch of images n.{idx} |'
                  f'epoch = {epoch:d}| accuracy = {acuracy:.5f}, dice = {dice_coef:.5f} '
                  f'jacaard_similarity = {jacaard_similarity:.5f}')

            # compute accuracy for all images in batch
            num_correct += (x == target).sum()
            num_pixels += torch.numel(x)

        print(f' Mean of metrics  | dice = {dice_score_dataset/len(loader):.5f}, '
              f'accuracy = {accuracy_dataset/len(loader):.5f}, '
              f'jacaard_similarity = {jacaaard_similarity_dataset/len(loader):.5f}')

        print(f"Correct px are {num_correct}/{num_pixels} with accuracy "
              f"{num_correct / num_pixels * 100:.2f} % for all images in batch")


        avg_val_loss = val_loss / len(loader)

    model.train()
    return avg_val_loss, valid_losses

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

def train_val_loss_as_graph(avg_train_loss, avg_val_loss, train_losses, val_losses, epoch, num_epochs)-> None:
    """
    Save train and validation losses as graph
    :param train_losses: train losses for epoch
    :param val_losses: validation losses for epoch
    :param epoch: actual epoch
    :param num_epochs: n. epochs to train
    :return: None
    """
    plt.figure(figsize=(10, 6))
    smoothing_factor = 3

    smoothed_train_losses = [
        sum(train_losses[i - smoothing_factor:i]) / smoothing_factor if i > smoothing_factor else sum(
            train_losses[:i + 1]) / (i + 1) for i in range(len(train_losses))]
    smoothed_valid_losses = [
        sum(val_losses[i - smoothing_factor:i]) / smoothing_factor if i > smoothing_factor else sum(
            val_losses[:i + 1]) / (i + 1) for i in range(len(val_losses))]

    plt.plot(smoothed_train_losses, label='Training Loss')
    plt.plot(smoothed_valid_losses, label='Validation Loss')

    # plt.plot(epoch, smoothed_train_losses, label='Training Loss')
    # plt.plot(epoch, smoothed_valid_losses, label='Validation Loss')

    # Highlight minimum validation loss
    min_valid_loss = min(smoothed_valid_losses)
    min_valid_loss_epoch = smoothed_valid_losses.index(min_valid_loss)
    plt.scatter(min_valid_loss_epoch, min_valid_loss, color='red', marker='o', label='Min Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, max(max(smoothed_train_losses), max(smoothed_valid_losses)) + 0.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists('.//loses//'):
        os.mkdir('.//loses//')
    plt.savefig(f'.//loses//combined_loss_{epoch}.png')
    plt.close()

    print(f'Epoch [{epoch}/{num_epochs}], '
          f'Training Loss: {avg_train_loss:.4f}, '
          f'Validation Loss: {avg_val_loss:.4f}, '
          f'Min Validation Loss at Epoch: {min_valid_loss_epoch} ({min_valid_loss:.4f})')
