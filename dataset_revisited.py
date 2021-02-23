# Essentials
import random
import os
from tifffile import TiffFile
from operator import itemgetter
# Data
import numpy as np
# Sklearn
from sklearn.model_selection import train_test_split
# Torch
import torch
from torch.utils.data import Dataset, Subset
# Code
from LCD import LandCoverData as LCD

class ImageSegementationDataset(Dataset):
    in_channels = 4
    out_channels = 10

    def __init__(self, images_dir, path_index=None, mode="train", random_state=42):

        assert mode in ["train", "valid", "test"], "mode should either be 'train' or 'test'"

        self.mode = mode
        self.images_dir = images_dir
        if self.mode == 'test':
            self.path_indices = os.listdir(os.path.join(self.images_dir, 'images'))
        else:
            self.path_indices = path_index
        self.random_state = random_state

    def transform(self, image, mask=None):

        if self.random_state:
            random.seed(self.random_state)

        if self.mode == 'train':
            if random.random() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            if random.random() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)

            if random.random() > 0.5:
                image = np.rot90(image)
                mask = np.rot90(mask)

            elif random.random() > 0.5:
                image = np.rot90(image, 3)
                mask = np.rot90(mask, 3)
            image = np.float32(image) / LCD.TRAIN_PIXELS_MAX

            image = torch.from_numpy(image.transpose(2, 0, 1).copy())
            mask = torch.from_numpy(mask.transpose(2, 0, 1).copy())
            return image, mask

        elif self.mode == 'valid':
            image = np.float32(image) / LCD.TRAIN_PIXELS_MAX
            image = torch.from_numpy(image.transpose(2, 0, 1).copy())
            mask = torch.from_numpy(mask.transpose(2, 0, 1).copy())
            return image, mask

        else:
            image = np.float32(image) / LCD.TRAIN_PIXELS_MAX
            image = torch.from_numpy(image.transpose(2, 0, 1).copy())
            return image          

    @staticmethod
    def parse_tiff(path):
        """Loads a Tiff image 
        Args:
            image_path (bytes): path to image
        Returns:
            numpy.array[uint16]: the image 
        """
        with TiffFile(path) as tifi:
            image = tifi.asarray()
        return image

    def __len__(self):
        return len(self.path_indices)

    def __getitem__(self, idx):

        if self.mode == 'train' or self.mode == 'valid':
            image_path = os.path.join(self.images_dir, 'images', self.path_indices[idx])
            mask_path = os.path.join(self.images_dir, 'masks', self.path_indices[idx])
            image = ImageSegementationDataset.parse_tiff(image_path)
            mask = ImageSegementationDataset.parse_tiff(mask_path)
            # add channel dimension to mask: (256, 256, 1)
            mask = np.expand_dims(mask, axis=-1)
            image, mask = self.transform(image, mask)
            return image, mask


        else:
            image_path = os.path.join(self.images_dir, 'images', self.path_indices[idx])
            print(self.path_indices[idx])
            image = ImageSegementationDataset.parse_tiff(image_path)
            image = self.transform(image)
            return image

def train_val_dataset(train_dir, val_split=0.25):
    list_images = os.listdir(os.path.join(train_dir, 'images'))
    number_files = len(list_images)
    train_idx, val_idx = train_test_split(range(number_files), test_size=val_split)
    return itemgetter(*train_idx)(list_images), itemgetter(*val_idx)(list_images)