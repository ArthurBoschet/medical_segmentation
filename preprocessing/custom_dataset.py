import os
import torch
import numpy as np

from torch.utils.data import Dataset


class MedicalImageDataset(Dataset):
    def __init__(self, path, resize=None):
        '''
        Initialize the custom dataset for segmentation of 3D medical images
        
        Args:
            path: str
                Path to the folder containing the data for the task
            resize: tuple(int, int, int)
                Size to resize the images and labels to
                Remember: (depth, height, width)
        '''
        self.path = path
        self.resize = resize

    def __len__(self):
        return len(os.listdir(self.path))//2

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
        '''
        # get image and label
        image = np.load(os.path.join(self.path, f"image_{str(idx).zfill(3)}.npy"))
        label = np.load(os.path.join(self.path, f"label_{str(idx).zfill(3)}.npy"))

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

        return image, label
