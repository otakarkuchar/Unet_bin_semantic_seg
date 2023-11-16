import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join



# get all paths from folder
directory_masks = 'input_data/cloud_masks/'
masks = [f for f in listdir(directory_masks) if isfile(join(directory_masks, f))]

directory_img = 'input_data/subscenes/'
# directory_img = 'C:/Users/Tescan_lab/Downloads/4172871/subscenes'
images = [f for f in listdir(directory_img) if isfile(join(directory_img, f))]

for file_masks, file_images in zip(masks, images):
    mask_name = directory_masks + file_masks
    mask = np.load(mask_name)
    # img[:, :, 1] = False
    mask = mask*255
    plt.imshow(mask)
    plt.show()

    img_name = directory_img + file_images
    img = np.load(img_name)
    # img[:, :, 1] = False
    img = img*255
    plt.imshow(img)
    plt.show()



    # file_name_img = directory_img + file_name.replace('.npy', '.png')
    # img = cv2.imread(file_name_img)
    # plt.imshow(img)
    # plt.show()



