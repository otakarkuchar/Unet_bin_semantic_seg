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
    def __init__(self, image_dir: str, mask_dir: str, is_train: bool, image_height: int,
                 image_width: int, apply_transform: bool = True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.image_height = image_height
        self.image_width = image_width
        self.apply_transform = apply_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path: str = os.path.join(self.mask_dir, self.images[index])
        image = np.array(np.load(img_path), dtype=np.float32)

        # plt.imshow(image[:, :, :3])
        # plt.show()

        mask = np.array(np.load(mask_path)*1.0, dtype=np.float32)
        # plt.imshow(mask)
        # plt.show()

        # CLEAR 0 R, CLOUD 1 G, CLOUD_SHADOW 2 B - > CLEAR 0 R
        # but for this assignment classify just into two classes and consider CLOUD_SHADOW as CLEAR.
        mask_bin = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.float32)
        mask_bin[np.where(mask[:, :, 1])] = 1.0
        mask = mask_bin.copy()
        del mask_bin
        # plt.imshow(mask_bin, cmap='gray')
        # plt.show()

        image_selected_befor = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        image_selected_befor[:, :, 0] = image[:, :, 0]
        image_selected_befor[:, :, 1] = image[:, :, 1]
        image_selected_befor[:, :, 2] = image[:, :, 2]
        image_selected_befor[:, :, 3] = image[:, :, 7]

        out = cv2.normalize(image_selected_befor[:,:,:3], None, alpha=0, beta=1,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # fig, ax = plt.subplots(1, 2, figsize=(15, 8))
        # ax[0].imshow(image_selected_befor[:, :, :3])
        # ax[0].set_title('Image before normalization')
        # ax[1].imshow(out[:, :, :3])
        # ax[1].set_title('Image after normalization')
        # plt.show()

        image_selected = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        image_selected[:, :, 0] = out[:, :, 0]
        image_selected[:, :, 1] = out[:, :, 1]
        image_selected[:, :, 2] = out[:, :, 2]
        image_selected[:, :, 3] = image[:, :, 7]

        # plt.imshow(image_selected[:, :, :3])



        image = image_selected.copy()
        del image_selected

        if self.is_train:
            train_transform = A.Compose(
                    [
                        A.Resize(height=self.image_height, width=self.image_width),
                        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=1),
                        # A.ColorJitter(brightness=(1.1, 1.2), p=1),
                        # A.ColorJitter(contrast=(1.1, 1.4), p=1),
                        # A.ColorJitter(saturation=(1,1.1), p=1),
                        # A.ColorJitter(brightness=0, contrast=(1, 5), saturation=0, hue=0),
                        # A.Posterize(num_bits=(0, 5), p=0.7),
                        A.Rotate(limit=35, p=1),
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=1),
                        # A.Normalize(
                        #     mean=(0.5, 0.5, 0.5),
                        #     std=(0.5, 0.5, 0.5),
                        #     max_pixel_value=1.0,
                        #     p=1.0),
                        ToTensorV2(),
                    ],
                )
        else:
            train_transform = A.Compose(
                    [
                        A.Resize(height=self.image_height, width=self.image_width),
                        # A.Normalize(
                        #     mean=(0.5, 0.5, 0.5),
                        #     std=(0.5, 0.5, 0.5),
                        #     max_pixel_value=1.0,
                        #     p=1.0),
                        ToTensorV2(),
                    ],
                )

        if self.apply_transform is not None:
            augmented = train_transform(image=image, mask=mask)
            image = augmented['image']
            mask_as_mask = augmented['mask']
            mask_as_mask = mask_as_mask.unsqueeze(0)

            mask_no_aug = mask
            image_no_aug = image

            mask_alone = train_transform(image=mask)
            mask_as_image = mask_alone['image']


            # fig, ax = plt.subplots(1, 3, figsize=(15, 8))
            # # mask_show = torch.concat((mask, mask[0:1, :, :]*0), dim=0)
            # ax[0].imshow(mask)
            # ax[1].imshow(mask_as_mask.cpu().permute(1, 2, 0).detach().numpy())
            # ax[2].imshow(mask_as_image.cpu().permute(1, 2, 0).detach().numpy())
            #
            # plt.show()

            # fig, ax = plt.subplots(2, 2, figsize=(15, 8))
            # # mask_show = torch.concat((mask, mask[0:1, :, :]*0), dim=0)
            # ax[0, 0].imshow(image_selected[:, :, :3])
            # ax[0, 0].set_title('Image before augmentation')
            # ax[0, 1].imshow(mask)
            # ax[0, 1].set_title('Mask before augmentation')
            # ax[1, 0].imshow(image[:3].cpu().permute(1, 2, 0).detach().numpy())
            # ax[1, 0].set_title('Image after augmentation')
            # ax[1, 1].imshow(mask.cpu().detach().numpy().astype(np.float32))
            # ax[1, 1].imshow(mask)
            # ax[1, 1].set_title('Mask after augmentation')
            # plt.show()

        return image, mask_as_mask
