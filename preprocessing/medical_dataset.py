import os
import torch
import torchio as tio
import numpy as np

from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from utils.data_utils import normalize_3d_array
from sklearn.model_selection import train_test_split


class MedicalImageDataset(Dataset):
    def __init__(self, path, num_classes=2, train=True, train_only=False, normalize=True, resize=None, transform=None):
        '''
        Initialize the custom dataset for segmentation of 3D medical images
        
        Args:
            path: str
                Path to the folder containing the data for the task
            num_classes: int
                Number of classes in the dataset
            train: bool
                Whether to load the train or validation split
            train_only: bool
                Whether to load the train split only
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

        # get the filenames of the images and labels
        filenames = sorted(os.listdir(path))
        images_filenames = filenames[:len(filenames)//2]
        labels_filenames = filenames[len(filenames)//2:]

        # load the train split only
        if train_only:
            self.images_files = images_filenames
            self.labels_files = labels_filenames
            return
        
        # split the data into train and validation
        train_images_files, val_images_files, train_labels_files, val_labels_files = train_test_split(
            images_filenames, 
            labels_filenames, 
            test_size=0.2,
            random_state=42
        )

        # load the train or validation split
        if train:
            self.images_files = train_images_files
            self.labels_files = train_labels_files
        else:
            self.images_files = val_images_files
            self.labels_files = val_labels_files

    def __len__(self):
        return len(self.images_files)

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
        # get image and label names
        image_name = self.images_files[idx]
        label_name = self.labels_files[idx]

        # get image and label numpy arrays
        image = np.load(os.path.join(self.path, image_name))
        label = np.load(os.path.join(self.path, label_name))

        # apply normalization if necessary
        if self.normalize:
            for c in range(image.shape[0]):
                image[c] = normalize_3d_array(image[c])

        # convert to torch tensors
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label)

        # add batch dimension
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)

        # adapt the number of channels in the label based on the number of classes from (N, 1, D, H, W) to (N, C, D, H, W)
        label = torch.cat([label == i for i in range(self.num_classes)], dim=1).float()

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

        # round label to int values
        label = torch.round(label).long()

        return image, label


class KFoldMedicalImageDataset(MedicalImageDataset):
    def __init__(self, path, k_folds, fold, train=True, *args, **kwargs):
        '''
        Initialize the KFoldMedicalImageDataset for k-fold cross-validation
        
        Args:
            path: str
                Path to the folder containing the data for the task
            k_folds: int
                Number of folds for cross-validation
            fold: int
                Current fold (0-indexed) for cross-validation
            train: bool
                Whether to load the train or validation split
            *args, **kwargs:
                Additional arguments to pass to the MedicalImageDataset constructor
        '''
        super().__init__(path, *args, **kwargs)
        self.path = path
        self.k_folds = k_folds
        self.fold = fold
        self.train = train

        filenames = sorted(os.listdir(path))
        images_filenames = filenames[:len(filenames)//2]
        labels_filenames = filenames[len(filenames)//2:]
        
        num_samples = len([f for f in os.listdir(self.path) if f.startswith("image")])
        self.indices = list(range(num_samples))

        self.kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        self.train_indices, self.val_indices = list(self.kfold.split(self.indices))[self.fold]

        if self.train:
            self.images_files = [images_filenames[i] for i in self.train_indices]
            self.labels_files = [labels_filenames[i] for i in self.train_indices]
        else:
            self.images_files = [images_filenames[i] for i in self.val_indices]
            self.labels_files = [labels_filenames[i] for i in self.val_indices]

    def __len__(self):
        return super().__len__()

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
        return super().__getitem__(idx)


class MedicalImageDatasetTest(Dataset):
    def __init__(self, path, normalize=True, resize=None, transform=None):
        '''
        Initialize the custom dataset for segmentation of 3D medical images
        
        Args:
            path: str
                Path to the folder containing the test data for the task
            normalize: bool
                Whether to normalize the images and labels between 0 and 1
            resize: tuple(int, int, int)
                Size to resize the images and labels to
                Remember: (depth, height, width)
            transform: torchvision.transforms
                Transformations to apply to the images and labels
        '''
        self.path = path
        self.normalize = normalize
        self.resize = resize
        self.transform = transform

        self.images_files = sorted(os.listdir(path))

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        '''
        Get the image and label at the given index
        
        Args:
            idx: int
                Index of the image and label to get
                
        Returns:
            image: torch.Tensor
                Tensor containing the image
            --> tensor has shape (channels, depth, height, width)
        '''
        # get image name
        image_name = self.images_files[idx]

        # get image numpy array
        image = np.load(os.path.join(self.path, image_name))

        # apply normalization if necessary
        if self.normalize:
            for c in range(image.shape[0]):
                image[c] = normalize_3d_array(image[c])

        # convert to torch tensor
        image = torch.from_numpy(image).float()

        # add batch dimension
        image = image.unsqueeze(0)

        # resize tensor if necessary
        if self.resize is not None:
            image = torch.nn.functional.interpolate(image, size=self.resize, mode="trilinear")

        # remove batch dimension
        image = image.squeeze(0)

        # apply transformations if necessary
        if self.transform is not None:
            subject = tio.Subject(
                image=tio.Image(tensor=image, type=tio.INTENSITY)
            )
            subject = self.transform(subject)
            image = subject.image.data

        return image