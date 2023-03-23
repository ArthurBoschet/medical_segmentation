import os
import shutil
import numpy as np
import nibabel as nib

from tqdm import tqdm
from sklearn.model_selection import train_test_split


def split_images_and_labels(input_folder):
    '''
    Split the images and labels from .npz files into separate .npy (numpy) files
    
    Args:
        input_folder: str 
            Path to the folder containing the images and labels
    '''
    image_dir = os.path.join(input_folder, 'imagesTr')
    label_dir = os.path.join(input_folder, 'labelsTr')
    for file in tqdm(os.listdir(image_dir)):
        if file.endswith('.npz'):
            data = np.load(os.path.join(image_dir, file))['data']
            idx = file.find('_')+1
            np.save(os.path.join(image_dir, f'image_{file[idx:idx+3]}.npy'), data[0])
            np.save(os.path.join(label_dir, f'label_{file[idx:idx+3]}.npy'), data[1])
            os.remove(os.path.join(image_dir, file))

def convert_niigz_to_numpy(input_folder):
    '''
    Convert the images and labels from .nii.gz files into .npy (numpy) files
    
    Args:
        input_folder: str
            Path to the folder containing the images and labels
    '''
    for subdir in ['imagesTr', 'labelsTr', 'imagesTs']:
        dir = os.path.join(input_folder, subdir)
        if os.path.exists(dir):
            for file in tqdm(os.listdir(dir)):
                if file.endswith('.nii.gz'):
                    data = nib.load(os.path.join(dir, file)).get_fdata()
                    data = np.transpose(data, (2, 0, 1))
                    idx = file.find('_')+1
                    np.save(os.path.join(dir, f'{subdir[:-3]}_{file[idx:idx+3]}.npy'), data)
                    os.remove(os.path.join(dir, file))

def convert_to_numpy(input_folder):
    '''
    Convert the images and labels into numpy arrays
    
    Args:
        input_folder: str 
            Path to the folder containing the images and labels
    '''
    image_dir = os.path.join(input_folder, 'imagesTr')
    if os.listdir(image_dir)[0].endswith('.npz'):
        split_images_and_labels(input_folder)
    elif os.listdir(image_dir)[0].endswith('.nii.gz'):
        convert_niigz_to_numpy(input_folder)
    else:
        raise ValueError('Images format not recognized (should be .npz or .nii.gz)')
    
def normalize_3d_array(array):
    '''
    Normalize a 3D array
    
    Args:
        array: np.ndarray
            Array to normalize
    
    Returns:
        array_norm: np.ndarray
            Normalized array
    '''
    # normalize the image
    array_norm = (array - array.mean()) / array.std()

    # rescale the image to [0, 1]
    array_norm = (array_norm - np.min(array_norm)) / (np.max(array_norm) - np.min(array_norm))
    
    return array_norm

def prepare_dataset_for_training(dataset_folder_path, output_dataset_path, val_size=0.2):
    '''
    Prepare the dataset for training by splitting it into training and validation sets for the dataloaders

    Args:
        dataset_folder_path: str
            Path to the dataset folder
        output_dataset_path: str
            Path to the output dataset folder that will be created and used for training
        val_size: float
            Size of the validation set
    '''
    # copy dataset from drive to virtual machine local disk
    shutil.copytree(dataset_folder_path, output_dataset_path)

    # split the data into training and validation sets
    train_images_files, val_images_files, train_labels_files, val_labels_files = train_test_split(
        os.listdir(os.path.join(output_dataset_path, "imagesTr")), 
        os.listdir(os.path.join(output_dataset_path, "labelsTr")), 
        test_size=val_size, 
        random_state=42
    )

    # create train and val folders
    os.makedirs(os.path.join(output_dataset_path, "train"))
    os.makedirs(os.path.join(output_dataset_path, "val"))
    os.makedirs(os.path.join(output_dataset_path, "test"))

    # copy validation images and labels to val folder
    for val_im_file, val_label_file in zip(val_images_files, val_labels_files):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTr", val_im_file), os.path.join(output_dataset_path, "val", val_im_file))
        shutil.copyfile(os.path.join(dataset_folder_path, "labelsTr", val_label_file), os.path.join(output_dataset_path, "val", val_label_file))
    # copy training images and labels to train folder
    for tr_im_file, tr_label_file in zip(train_images_files, train_labels_files):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTr", tr_im_file), os.path.join(output_dataset_path, "train", tr_im_file))
        shutil.copyfile(os.path.join(dataset_folder_path, "labelsTr", tr_label_file), os.path.join(output_dataset_path, "train", tr_label_file))
    # copy test images to test folder
    for test_im_file in os.listdir(os.path.join(dataset_folder_path, "imagesTs")):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTs", test_im_file), os.path.join(output_dataset_path, "test", test_im_file))

    # renames train files
    for i, (im_tr, la_tr) in enumerate(zip(sorted(train_images_files), sorted(train_labels_files))):
        os.rename(os.path.join(output_dataset_path, "train", im_tr), os.path.join(output_dataset_path, "train", f"image_{str(i).zfill(3)}.npy"))
        os.rename(os.path.join(output_dataset_path, "train", la_tr), os.path.join(output_dataset_path, "train", f"label_{str(i).zfill(3)}.npy"))
    # renames val files
    for i, (im_val, la_val) in enumerate(zip(sorted(val_images_files), sorted(val_labels_files))):
        os.rename(os.path.join(output_dataset_path, "val", im_val), os.path.join(output_dataset_path, "val", f"image_{str(i).zfill(3)}.npy"))
        os.rename(os.path.join(output_dataset_path, "val", la_val), os.path.join(output_dataset_path, "val", f"label_{str(i).zfill(3)}.npy"))
    # renames test files
    for i, test_im in enumerate(sorted(os.listdir(os.path.join(output_dataset_path, "test")))):
        os.rename(os.path.join(output_dataset_path, "test", test_im), os.path.join(output_dataset_path, "test", f"image_{str(i).zfill(3)}.npy"))

    # remove useless folders
    shutil.rmtree(os.path.join(output_dataset_path, "imagesTr"))
    shutil.rmtree(os.path.join(output_dataset_path, "labelsTr"))
    shutil.rmtree(os.path.join(output_dataset_path, "imagesTs"))
