import os

from torch.utils.data import DataLoader
from custom_dataset import MedicalImageDataset


def load_data(data_folder_path,
              batch_size=1, 
              shuffle=True,
              resize=None):
    '''
    Load the data for the task into pytorch DataLoaders

    Args:
        data_folder_path: str
            Path to the folder containing the data preprocessed for the task (images and labels)
        batch_size: int
            Batch size for the DataLoader
        shuffle: bool
            Whether to shuffle the data in the DataLoader
        resize: tuple
            Size to resize the images and labels to
            Remember: (depth, height, width)
        
    Returns:
        train_dataloader: DataLoader
            DataLoader for the training set
        val_dataloader: DataLoader
            DataLoader for the validation set
    '''
    # create the pytorch dataset
    train_dataset = MedicalImageDataset(os.path.join(data_folder_path, 'train'), resize=resize)
    val_dataset = MedicalImageDataset(os.path.join(data_folder_path, 'val'), resize=resize)

    # create the pytorch dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader
