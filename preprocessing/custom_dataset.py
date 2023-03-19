import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms


class MedicalImageDataset(Dataset):
    def __init__(self, images, labels, resize=None):
        '''
        Initialize the custom dataset for segmentation of 3D medical images
        
        Args:
            images: Iterable(numpy.ndarray)
                Iterable of numpy arrays containing the images
            labels: Iterable(numpy.ndarray)
                Iterable of numpy arrays containing the labels
            resize: tuple(int, int, int)
                Size to resize the images and labels to
                Remember: (depth, height, width)
        '''
        self.images = images
        self.labels = labels
        self.resize = resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Get the image and label at the given index
        
        Args:
            idx: int
                Index of the image and label to get
                
        Returns:
            image: torch.Tensor
                Tensor containing the image
            label: torch.Tensor
                Tensor containing the label
            --> both tensors have shape (channels, depth, height, width)
            idx: int
                Index of the item
        '''
        # get image and label
        image = self.images[idx]
        label = self.labels[idx]

        # convert to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        # add channel dimension
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)

        # add batch dimension
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)

        # resize tensors if necessary
        if self.resize is not None:
            image = torch.nn.functional.interpolate(image, size=self.resize, mode="trilinear")
            label = torch.nn.functional.interpolate(label, size=self.resize, mode="trilinear")

        # remove batch dimension
        image = image.squeeze(0)
        label = label.squeeze(0)

        # round label to 0 or 1
        label = torch.round(label)

        # replace -1 with 0
        label[label == -1] = 0

        # TODO: implement data augmentation

        return image, label, idx
