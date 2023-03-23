import os
import torch
import torchio as tio
import numpy as np

from torch.utils.data import Dataset
from utils.data_utils import normalize_3d_array


class MedicalImageDataset(Dataset):
    def __init__(self, path, num_classes=2, normalize=False, resize=None, transform=None):
        '''
        Initialize the custom dataset for segmentation of 3D medical images
        
        Args:
            path: str
                Path to the folder containing the data for the task
            num_classes: int
                Number of classes in the dataset
            normalize: bool
                Whether to normalize the images and labels between 0 and 1
            resize: tuple(int, int, int)
                Size to resize the images and labels to
                Remember: (depth, height, width)
            transform: torchvision.transforms
                Transformations to apply to the images and labels
        '''
        self.path = path
        self.num_classes = num_classes
        self.normalize = normalize
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len([f for f in os.listdir(self.path) if f.startswith("image")])

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

        # apply normalization if necessary
        if self.normalize:
            image = normalize_3d_array(image)

        # convert to torch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label)

        # add channel dimension
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)

        # add batch dimension
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)

        # adapt the number of channels in the label based on the number of classes from (N, 1, D, H, W) to (N, C, D, H, W)
        label = torch.cat([label == i for i in range(self.num_classes)], dim=1).long()

        # resize tensors if necessary
        if self.resize is not None:
            image = torch.nn.functional.interpolate(image, size=self.resize, mode="trilinear")
            label = torch.nn.functional.interpolate(label, size=self.resize, mode="trilinear")

        # remove batch dimension
        image = image.squeeze(0)
        label = label.squeeze(0)

        # apply transformations if necessary
        if self.transform is not None:
            subject = tio.Subject(
                image=tio.Image(tensor=image, type=tio.INTENSITY),
                label=tio.Image(tensor=label, type=tio.LABEL)
            )
            subject = self.transform(subject)
            image = subject.image.data
            label = subject.label.data

        # round label to 0 or 1
        label = torch.round(label)

        # replace -1 with 0 if any
        label[label == -1] = 0

        return image, label
