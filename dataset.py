import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class CloudDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, is_trained: bool, image_height: int,
                 image_width: int, apply_transform: bool = True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_trained = is_trained
        self.image_height = image_height
        self.image_width = image_width
        self.apply_transform = apply_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path: str = os.path.join(self.mask_dir, self.images[index])
        image_raw = np.array(np.load(img_path), dtype=np.float32)
        mask_raw = np.array(np.load(mask_path)*1.0, dtype=np.float32)

        # CLEAR 0 R, CLOUD 1 G, CLOUD_SHADOW 2 B - consider CLOUD_SHADOW as CLEAR.
        mask_gt = np.zeros((mask_raw.shape[0], mask_raw.shape[1]), dtype=np.float32)
        mask_gt[np.where(mask_raw[:, :, 1])] = 1.0

        image_normalized = cv2.normalize(image_raw[:,:,:3], None, alpha=0, beta=1,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        image_selected_norm = np.zeros((image_raw.shape[0], image_raw.shape[1], 4), dtype=np.float32)
        image_selected_norm[:, :, :3] = image_normalized[:, :, :3]
        image_selected_norm[:, :, 3] = image_raw[:, :, 7]

        if self.is_trained:
            train_transform = A.Compose(
                    [
                        A.Resize(height=self.image_height, width=self.image_width),
                        A.Rotate(limit=35, p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        ToTensorV2(),
                    ],
                )
        else:
            train_transform = A.Compose(
                    [
                        A.Resize(height=self.image_height, width=self.image_width),
                        ToTensorV2(),
                    ],
                )

        if self.apply_transform is not None:
            augmented = train_transform(image=image_selected_norm, mask=mask_gt)
            image_selected_norm = augmented['image']
            mask_gt = augmented['mask']
            mask_gt = mask_gt.unsqueeze(0)

        return image_selected_norm, mask_gt
