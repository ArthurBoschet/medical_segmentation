import os
import shutil

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from nnUNet.nnunet.experiment_planning.nnUNet_convert_decathlon_task import main as convert_decathlon_task
from nnUNet.nnunet.experiment_planning.nnUNet_plan_and_preprocess import main as preprocess_task
from custom_dataset import MedicalImageDataset
from utils import convert_to_numpy, add_padding, get_data


def load_data(task_folder_path, 
              dataset_type="raw", 
              stage="stage_0",
              val_size=0.2, 
              batch_size=1, 
              shuffle=True):
    '''
    Load the data for the task into pytorch DataLoaders

    Args:
        task_folder_path: str
            Path to the folder containing the data for the task
        dataset_type: str
            Type of dataset to load (raw, cropped, preprocessed, preprocessed_pad)
        stage: str
            Stage of preprocessing to load (stage_0, stage_1)
            Only applicable if dataset_type is preprocessed or preprocessed_pad
        val_size: float
            Proportion of the data to use for validation
        batch_size: int
            Batch size for the DataLoader
        shuffle: bool
            Whether to shuffle the data in the DataLoader
        
    Returns:
        train_loader: DataLoader
            DataLoader for the training set
        val_loader: DataLoader
            DataLoader for the validation set
    
    Final folder structure:

    task_folder_path
    ├── raw
    │   ├── imagesTr
    │   │   ├── image_000.npy
    │   │   ├── ...
    │   ├── labelsTr
    │   │   ├── label_000.npy
    │   │   ├── ...
    │   ├── imagesTs
    │   │   ├── image_000.npy
    │   │   ├── ...
    ├── cropped
    │   ├── imagesTr
    │   │   ├── image_000.npy
    │   │   ├── ...
    │   ├── labelsTr
    │   │   ├── label_000.npy
    │   │   ├── ...
    ├── preprocessed
    │   ├── stage_0
    │   │   ├── imagesTr
    │   │   │   ├── image_000.npy
    │   │   │   ├── ...
    │   │   ├── labelsTr
    │   │   │   ├── label_000.npy
    │   │   │   ├── ...
    │   ├── stage_1
    │   │   ├── imagesTr
    │   │   │   ├── image_000.npy
    │   │   │   ├── ...
    │   │   ├── labelsTr
    │   │   │   ├── label_000.npy
    │   │   │   ├── ...
    ├── preprocessed_pad
    │   ├── stage_0
    │   │   ├── imagesTr
    │   │   │   ├── image_000.npy
    │   │   │   ├── ...
    │   │   ├── labelsTr
    │   │   │   ├── label_000.npy
    │   │   │   ├── ...
    ├── dataset.json
    '''
    assert dataset_type in ["raw", "cropped", "preprocessed", "preprocessed_pad"], "dataset_type must be one of raw, cropped, preprocessed, preprocessed_pad"
    assert stage in ["stage_0", "stage_1"], "stage must be one of stage_0, stage_1"
    # check if the data has already been preprocessed
    if not os.path.exists(os.path.join(task_folder_path, 'preprocessed')):
        preprocess_data(task_folder_path)

    # convert the images and labels into numpy arrays
    if not os.listdir(os.path.join(task_folder_path, 'raw', 'imagesTr'))[0].endswith('.npy'):
        convert_to_numpy(os.path.join(task_folder_path, 'raw'))
    if not os.listdir(os.path.join(task_folder_path, 'cropped', 'imagesTr'))[0].endswith('.npy'):
        convert_to_numpy(os.path.join(task_folder_path, 'cropped'))
    if os.path.exists(os.path.join(task_folder_path, 'preprocessed', 'stage_0', 'imagesTr')):
        if not os.listdir(os.path.join(task_folder_path, 'preprocessed', 'stage_0', 'imagesTr'))[0].endswith('.npy'):
            convert_to_numpy(os.path.join(task_folder_path, 'preprocessed', 'stage_0'))
    if os.path.exists(os.path.join(task_folder_path, 'preprocessed', 'stage_1', 'imagesTr')):
        if not os.listdir(os.path.join(task_folder_path, 'preprocessed', 'stage_1', 'imagesTr'))[0].endswith('.npy'):
            convert_to_numpy(os.path.join(task_folder_path, 'preprocessed', 'stage_1'))

    # add padding to the stage 0 preprocessed images and labels
    if not os.path.exists(os.path.join(task_folder_path, 'preprocessed_pad')):
        add_padding(os.path.join(task_folder_path, 'preprocessed', 'stage_0'))

    # get the data from the given dataset_type
    if dataset_type == "preprocessed" or dataset_type == "preprocessed_pad":
        images, labels = get_data(os.path.join(task_folder_path, dataset_type, stage))
    else:
        images, labels = get_data(os.path.join(task_folder_path, dataset_type))

    # split the data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=val_size, random_state=42)
    
    # create the pytorch dataset
    train_dataset = MedicalImageDataset(train_images, train_labels)
    val_dataset = MedicalImageDataset(val_images, val_labels)

    # create the pytorch dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader
    
def preprocess_data(task_folder_path):
    '''
    Convert the data into the nnUNet format and preprocess it
    
    Args:
        task_folder_path: str 
            Path to the task folder

    The folder structure should be:

    task_folder_path
    ├── imagesTr
    │   ├── la_000.nii.gz
    │   ├── ...
    ├── labelsTr
    │   ├── la_000.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── la_000.nii.gz
    │   ├── ...
    ├── dataset.json
    '''
    convert_decathlon_task(task_folder_path)
    preprocess_task(task_folder_path)

    # --> refactor structure

    # raw
    shutil.move(os.path.join(task_folder_path, 'preprocessed_data', 'raw'), os.path.join(task_folder_path))

    # cropped
    os.makedirs(os.path.join(task_folder_path, 'cropped', 'imagesTr'))
    os.makedirs(os.path.join(task_folder_path, 'cropped', 'labelsTr'))
    for file in os.listdir(os.path.join(task_folder_path, 'preprocessed_data', 'cropped')):
        if file.endswith('.npz'):
            shutil.move(os.path.join(task_folder_path, 'preprocessed_data', 'cropped', file), os.path.join(task_folder_path, 'cropped', 'imagesTr', file))
    
    # preprocessed -- stage 0
    if os.path.exists(os.path.join(task_folder_path, 'preprocessed_data', 'final', 'nnUNetData_plans_v2.1_stage0')):
        os.makedirs(os.path.join(task_folder_path, 'preprocessed', 'stage_0', 'imagesTr'))
        os.makedirs(os.path.join(task_folder_path, 'preprocessed', 'stage_0', 'labelsTr'))
        for file in os.listdir(os.path.join(task_folder_path, 'preprocessed_data', 'final', 'nnUNetData_plans_v2.1_stage0')):
            if file.endswith('.npz'):
                shutil.move(os.path.join(task_folder_path, 'preprocessed_data', 'final', 'nnUNetData_plans_v2.1_stage0', file), os.path.join(task_folder_path, 'preprocessed', 'stage_0', 'imagesTr', file))

    # preprocessed -- stage 1
    if os.path.exists(os.path.join(task_folder_path, 'preprocessed_data', 'final', 'nnUNetData_plans_v2.1_stage1')):
        os.makedirs(os.path.join(task_folder_path, 'preprocessed', 'stage_1', 'imagesTr'))
        os.makedirs(os.path.join(task_folder_path, 'preprocessed', 'stage_1', 'labelsTr'))
        for file in os.listdir(os.path.join(task_folder_path, 'preprocessed_data', 'final', 'nnUNetData_plans_v2.1_stage1')):
            if file.endswith('.npz'):
                shutil.move(os.path.join(task_folder_path, 'preprocessed_data', 'final', 'nnUNetData_plans_v2.1_stage1', file), os.path.join(task_folder_path, 'preprocessed', 'stage_1', 'imagesTr', file))

    # remove useless folders
    shutil.rmtree(os.path.join(task_folder_path, 'preprocessed_data'))
    shutil.rmtree(os.path.join(task_folder_path, 'imagesTr'))
    shutil.rmtree(os.path.join(task_folder_path, 'imagesTs'))
    shutil.rmtree(os.path.join(task_folder_path, 'labelsTr'))

    # remove useless files
    os.remove(os.path.join(task_folder_path, 'raw', 'dataset.json'))
