import os

from torch.utils.data import DataLoader
from medical_dataset import MedicalImageDataset


def load_data(data_folder_path,
              batch_size=1, 
              num_classes=2,
              shuffle=True,
              normalize=False,
              resize=None,
              transform=None):
    '''
    Load the data for the task into pytorch DataLoaders

    Args:
        data_folder_path: str
            Path to the folder containing the data preprocessed for the task (images and labels)
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
    train_dataset = MedicalImageDataset(os.path.join(data_folder_path, 'train'), num_classes=num_classes, normalize=normalize, resize=resize, transform=transform)
    val_dataset = MedicalImageDataset(os.path.join(data_folder_path, 'val'), num_classes=num_classes, normalize=normalize, resize=resize, transform=transform)
    test_dataset = MedicalImageDataset(os.path.join(data_folder_path, 'test'), num_classes=num_classes, normalize=normalize, resize=resize, transform=transform)

    # create the pytorch dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
