import matplotlib.pyplot as plt
import onnxruntime as rt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import CloudDataset

# Hyperparameters
_DEVICE: str = "cpu"

# Load the ONNX model
onnx_model_path = "interfer_model.onnx"
session = rt.InferenceSession(onnx_model_path)


def sigmoid(input_array: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function
    :type input_array: np.ndarray
    :param input_array: output from model
    :return: 
    """
    return 1/(1 + np.exp(-input_array))


_VALID_IMG_DIR: str = 'data/valid_images/'
_VALID_MASK_DIR: str = 'data/valid_masks/'
valid_set = CloudDataset(
    _VALID_IMG_DIR,
    _VALID_MASK_DIR,
    False,
    224,
    224,
    False)
train_loader = DataLoader(valid_set, batch_size=14, shuffle=True)
# input_data = valid_set[0][0]
for data in train_loader:
    img, mask = data


    if _DEVICE == "cpu":
        # img = img.unsqueeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
        # img = img.cpu().permute(2, 3, 0).detach().numpy()
        img = img.cpu().detach().numpy()
        img = torch.randn(20, 4, 224, 224).detach().cpu().numpy()
        mask = mask.cpu().detach().numpy()

    # Perform inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: img})
    result = result[0]
    for i in range(len(result)):
        show_result = sigmoid(result[i][0])
        target = mask[i][0]

        plt.imshow(show_result)
        plt.show()
        plt.imshow(target)
        plt.show()

        # Display the inference result
        print(f"Input Data: {img}")
        print(f"Output Data: {result[0]}")

