import os

from torch.utils.data import DataLoader
from medical_dataset import MedicalImageDataset, KFoldMedicalImageDataset, MedicalImageDatasetTest


def load_data(data_folder_path,
              batch_size=2, 
              num_classes=2,
              shuffle=True,
              normalize=True,
              resize=None,
              transform=None):
    '''
    Load the data for the task into pytorch DataLoaders

    Args:
        data_folder_path: str
            Path to the folder containing the train data for the task (images and labels)
        batch_size: int
            Batch size for the DataLoader
        num_classes: int
            Number of classes in the dataset
        shuffle: bool
            Whether to shuffle the data in the DataLoader
        normalize: bool
            Whether to normalize the images and labels between 0 and 1
        resize: tuple
            Size to resize the images and labels to
            Remember: (depth, height, width)
        transform: torchvision.transforms
            Transformations to apply to the images and labels
        
    Returns:
        train_dataloader: DataLoader
            DataLoader for the training set
        val_dataloader: DataLoader
            DataLoader for the validation set
    '''
    # create the pytorch dataset
    train_dataset = MedicalImageDataset(os.path.join(data_folder_path, 'train_val'), num_classes=num_classes, train=True, normalize=normalize, resize=resize, transform=transform)
    val_dataset = MedicalImageDataset(os.path.join(data_folder_path, 'train_val'), num_classes=num_classes, train=False, normalize=normalize, resize=resize, transform=transform)

    # create the pytorch dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # add custom keys
    train_dataloader.dataset.input_size = resize
    train_dataloader.dataset.dataset_task = data_folder_path.split('/')[-1]
    train_dataloader.dataset.num_classes = num_classes
    train_dataloader.dataset.shuffle = shuffle
    train_dataloader.dataset.normalize = normalize

    return train_dataloader, val_dataloader


def load_data_kfold(data_folder_path,
                    k_folds=5,
                    fold=0,
                    batch_size=2, 
                    num_classes=2,
                    shuffle=True,
                    normalize=True,
                    resize=None,
                    transform=None):
    '''
    Load the kth fold data into pytorch DataLoaders

    Args:
        data_folder_path: str
            Path to the folder containing the data preprocessed for the task (images and labels)
        k_folds: int
            Number of folds to use for k-fold cross validation
        fold: int
            Index of the fold to use for validation
        batch_size: int
            Batch size for the DataLoader
        num_classes: int
            Number of classes in the dataset
        shuffle: bool
            Whether to shuffle the data in the DataLoader
        normalize: bool
            Whether to normalize the images and labels between 0 and 1
        resize: tuple
            Size to resize the images and labels to
            Remember: (depth, height, width)
        transform: torchvision.transforms
            Transformations to apply to the images and labels
        
    Returns:
        train_dataloader: DataLoader
            DataLoader for the training set
        val_dataloader: DataLoader
            DataLoader for the validation set
    '''
    # create the pytorch dataset
    train_dataset = KFoldMedicalImageDataset(os.path.join(data_folder_path, 'train_val'), k_folds=k_folds, fold=fold, train=True, num_classes=num_classes, normalize=normalize, resize=resize, transform=transform)
    val_dataset = KFoldMedicalImageDataset(os.path.join(data_folder_path, 'train_val'), k_folds=k_folds, fold=fold, train=False, num_classes=num_classes, normalize=normalize, resize=resize, transform=transform)

    # create the pytorch dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # add custom keys
    train_dataloader.dataset.input_size = resize
    train_dataloader.dataset.dataset_task = data_folder_path.split('/')[-1]
    train_dataloader.dataset.num_classes = num_classes
    train_dataloader.dataset.shuffle = shuffle
    train_dataloader.dataset.normalize = normalize

    return train_dataloader, val_dataloader

def load_test_data(data_folder_path,
                   batch_size=1, 
                   shuffle=False,
                   normalize=True,
                   resize=None,
                   transform=None):
    '''
    Load the test data for the task into pytorch DataLoader

    Args:
        data_folder_path: str
            Path to the folder containing the test data for the task (images and labels)
        batch_size: int
            Batch size for the DataLoader
        num_classes: int
            Number of classes in the dataset
        shuffle: bool
            Whether to shuffle the data in the DataLoader
        normalize: bool
            Whether to normalize the images and labels between 0 and 1
        resize: tuple
            Size to resize the images and labels to
            Remember: (depth, height, width)
        transform: torchvision.transforms
            Transformations to apply to the images and labels
        
    Returns:
        test_dataloader: DataLoader
            DataLoader for the test set
    '''
    # create the pytorch dataset
    test_dataset = MedicalImageDatasetTest(os.path.join(data_folder_path, 'test'), normalize=normalize, resize=resize, transform=transform)

    # create the pytorch dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_dataloader