import torch
import numpy as np

from torch.utils.data import Dataset


class MedicalImageDataset(Dataset):
    def __init__(self, images, labels):
        '''
        Initialize the custom dataset for segmentation of 3D medical images
        
        Args:
            images: Iterable(numpy.ndarray)
                Iterable of numpy arrays containing the images
            labels: Iterable(numpy.ndarray)
                Iterable of numpy arrays containing the labels
        '''
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # convert to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label
