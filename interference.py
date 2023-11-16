import matplotlib.pyplot as plt
import onnxruntime as rt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset import CloudDataset

# Hyperparameters
_DEVICE: str = "cpu"
_IMG_DIR: str = 'data/valid_images/'
_MASK_DIR: str = 'data/valid_masks/'

# Load the ONNX model
onnx_model_path = "interfer_model.onnx"
session = rt.InferenceSession(onnx_model_path)


def sigmoid(input_array: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function
    :type input_array: np.ndarray
    :param input_array: output from model
    :return: input_data after sigmoid activation function
    """
    return 1/(1 + np.exp(-input_array))


valid_set = CloudDataset(
    image_dir=_IMG_DIR,
    mask_dir=_MASK_DIR,
    is_trained=False,
    image_width=224,
    image_height=224,
    apply_transform=False)

train_loader = DataLoader(valid_set, batch_size=14, shuffle=True)
for data in train_loader:
    img, mask = data

    if _DEVICE == "cpu":
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

