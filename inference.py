import matplotlib.pyplot as plt
import onnxruntime as rt
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import CloudDataset

# Hyperparameters
_PATH_TO_ONNX_MODEL: str = "my_model_checkpoint.onnx"
_TEST_IMG_DIR: str = 'data/valid_images/'
_TEST_MASK_DIR: str = 'data/valid_masks/'
_BATCH_SIZE_TEST_DATA: int = 16
_IMSHOW_RESULTS: bool = False
_SAVE_RESULTS: bool = True
_VERBOSE_METRICS: bool = True


def sigmoid(input_array: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function
    :type input_array: np.ndarray
    :param input_array: output from model
    :return: input_data after sigmoid activation function
    """
    score = 1/(1 + np.exp(-input_array))

    return score


def get_data(test_img_dir: str, test_mask_dir: str, batch_size: int) -> DataLoader:
    """
    Get test data
    :return: test data loader object
    """
    test_set = CloudDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        is_trained=False,
        image_width=224,
        image_height=224,
        apply_transform=False)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return test_loader

def dice(prediction: np.ndarray, target: np.ndarray, smooth=1.) -> float:
    """
    Dice score function
    :param prediction: prediction from model
    :param target: target from dataset
    :param smooth: smoothness
    :return:
    """
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    intersection = np.sum(prediction * target)
    dice = (2. * intersection + smooth) / (np.sum(prediction) + np.sum(target) + smooth)

    return dice


def show_results_or_save(show_original_img: np.ndarray, target_mask: np.ndarray, output: np.ndarray,
                 i: int, idx_batch: int) -> None:
    """
    Show results of inference
    :param show_original_img: original image
    :param target_mask: target mask
    :param output: output from model
    :param i: index of image in batch
    :param idx_batch: index of batch
    :return: None
    """
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(show_original_img)
    ax[0].set_title('Original image')
    ax[1].imshow(target_mask, cmap='gray', vmin=0.0, vmax=1.0)
    ax[1].set_title('Ground truth target')
    ax[2].imshow(output, cmap='gray', vmin=0.0, vmax=1.0)
    ax[2].set_title('Predicted')
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    if _SAVE_RESULTS:
        if not os.path.exists("results"):
            os.mkdir("results")
        plt.savefig(f"results/Result of inference {i} of "
                    f"batch {idx_batch + 1} of {len(test_loader)}.png")

    if _IMSHOW_RESULTS:
        plt.show()
        plt.pause(0.1)
    plt.close(fig)


def run_inference(test_loader: object) -> None:
    """
    Run inference on test data
    :param test_loader: test data loader object
    :return: None
    """
    dice_all_data = []
    for idx_batch, data in enumerate(test_loader):
        dice_batch = []
        input_data, target = data
        input_data = input_data.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        # Perform inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_data})[0]

        for i in range(len(result)):

            output = sigmoid(result[i][0])
            output = np.where(output > 0.5, 1.0, 0.0).astype(np.float32)
            target_mask = target[i][0].astype(np.float32)
            show_original_img = np.transpose(input_data[i][:3], (1, 2, 0))

            if _IMSHOW_RESULTS or _SAVE_RESULTS:
                show_results_or_save(show_original_img, target_mask, output, i, idx_batch)

            if _VERBOSE_METRICS:
                dice_score = dice(output, target_mask)
                # print(f" Dice score for interference {i} of batch "
                #       f"{idx_batch+1} of {_BATCH_SIZE_TEST_DATA} is {dice_score}")
                dice_all_data.append(dice_score)
                dice_batch.append(dice_score)

        print(f"Batch {idx_batch+1} of {len(test_loader)} done.")

        if _VERBOSE_METRICS:
            print(f"Mean dice score for batch {idx_batch+1} is {np.mean(dice_batch)}")

    if _VERBOSE_METRICS:
        print("=====================================================")
        print("Summary: ")
        print(f"Total number of batches is {len(test_loader)}")
        print(f"Total number of images is {len(dice_all_data)}")
        print(f"Mean dice score for all data is {np.mean(dice_all_data)}")
        print("=====================================================")


if __name__ == '__main__':
    session = rt.InferenceSession(_PATH_TO_ONNX_MODEL)
    test_loader = get_data(_TEST_IMG_DIR, _TEST_MASK_DIR, _BATCH_SIZE_TEST_DATA)
    run_inference(test_loader)


