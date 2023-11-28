import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torchvision
from pathlib import Path


class CloudDatasetKaggle(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        super().__init__()

        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch

    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):

        files = {'red': r_file,
                 'green': g_dir / r_file.name.replace('red', 'green'),
                 'blue': b_dir / r_file.name.replace('red', 'blue'),
                 'nir': nir_dir / r_file.name.replace('red', 'nir'),
                 'gt': gt_dir / r_file.name.replace('red', 'gt')}

        return files

    def __len__(self):

        return len(self.files)

    def open_as_array(self, idx, invert=False, include_nir=False):

        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                            ], axis=2)

        raw_rgb = cv2.normalize(raw_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # raw_rgb.resize((224, 224, 3), refcheck=False)
        # raw_rgb = cv2.normalize(raw_rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # plt.imshow(raw_rgb)
        # plt.show()


        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)

            nir = cv2.normalize(nir, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            nir = np.expand_dims(nir, 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
            # raw_rgb = np.resize(raw_rgb, (224, 224))
            raw_rgb = cv2.resize(raw_rgb, (224,224), interpolation=cv2.INTER_AREA)
            # plt.imshow(raw_rgb[:, :, :3])
            # plt.show()

        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))

        # plt.imshow(raw_rgb[:3,:,:].transpose((1, 2, 0)))
        # plt.show()

        # normalize
        return raw_rgb

    def open_mask(self, idx, add_dims=False):

        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask == 255, 1, 0)
        raw_mask = cv2.resize(raw_mask.astype(np.float32), (224, 224), interpolation=cv2.INTER_AREA)

        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):

        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        # x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        # x = x.resize_(4, 224, 224)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.float32).unsqueeze(0)
        # y = y.resize_(1, 224, 224)

        # fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        # ax[0, 0].imshow(x[0:3, :, :].permute(1, 2, 0).cpu().detach().numpy())
        # ax[1, 0].imshow(x[1:4, :, :].permute(1, 2, 0).cpu().detach().numpy())
        # ax[0, 0].set_title('Image after augmentation')
        # ax[0, 1].imshow(y[0].cpu().detach().numpy())
        # plt.show()

        return x, y

    def open_as_pil(self, idx):

        arr = 256 * self.open_as_array(idx)

        return Image.fromarray(arr.astype(np.uint8), 'RGB')

    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())

        return s

