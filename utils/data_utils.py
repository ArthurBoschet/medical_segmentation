import os
import json
import shutil
import numpy as np
import nibabel as nib

from tqdm import tqdm


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
            for file in tqdm(sorted(os.listdir(dir))):
                if file.endswith('.nii.gz'):
                    data = nib.load(os.path.join(dir, file)).get_fdata()
                    if data.ndim == 3:
                        data = np.transpose(data, (2, 0, 1))
                    elif data.ndim == 4:
                        data = np.transpose(data[:, :, :, 0], (2, 0, 1))
                    idx_start = file.find('_')+1
                    idx_end = file.find('.nii.gz')
                    np.save(os.path.join(dir, f'{subdir[:-3]}_{file[idx_start:idx_end].zfill(3)}.npy'), data)
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

    # paths
    train_images_files = sorted(os.listdir(os.path.join(output_dataset_path, "imagesTr")))
    train_labels_files = sorted(os.listdir(os.path.join(output_dataset_path, "labelsTr")))

    # create train, val and test folders
    os.makedirs(os.path.join(output_dataset_path, "train_val"))
    os.makedirs(os.path.join(output_dataset_path, "test"))

    # copy training images and labels to train_val folders
    for tr_im_file, tr_label_file in zip(train_images_files, train_labels_files):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTr", tr_im_file), os.path.join(output_dataset_path, "train_val", tr_im_file))
        shutil.copyfile(os.path.join(dataset_folder_path, "labelsTr", tr_label_file), os.path.join(output_dataset_path, "train_val", tr_label_file))
    # copy test images to test folder
    for test_im_file in os.listdir(os.path.join(dataset_folder_path, "imagesTs")):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTs", test_im_file), os.path.join(output_dataset_path, "test", test_im_file))
    
    # remove useless folders
    shutil.rmtree(os.path.join(output_dataset_path, "imagesTr"))
    shutil.rmtree(os.path.join(output_dataset_path, "labelsTr"))
    shutil.rmtree(os.path.join(output_dataset_path, "imagesTs"))

def save_nifti(image, affine, filename):
    '''
    Save a 3D image to a nifti file.
    
    Args:
        image (np.array): 
            3D image
        affine (np.array):
            Affine matrix
        filename (str):
            Path to the nifti file
    '''
    img = nib.Nifti1Image(image, affine)
    nib.save(img, filename)


def reconstruct_affine_matrix(header_file_path):
    '''
    Reconstruct the affine matrix from the header of a nifti file.

    Args:
        header_file_path (str):
            Path to the nifti file

    Returns:
        affine (np.array):
            Affine matrix
    '''
    # load the header
    header = nib.load(header_file_path).header

    # voxel spacing
    dx, dy, dz = header.get_zooms()

    # origin of the image coordinate system
    x0, y0, z0 = header.get_qform()[:3, 3]

    # reconstruction of the affine matrix
    affine = np.array([[dx, 0, 0, x0],
                       [0, dy, 0, y0],
                       [0, 0, dz, z0],
                       [0, 0, 0, 1]])
    
    return affine


def get_original_shape(image_file_path):
    '''
    Get the original shape of a nifti file.

    Args:
        image_file_path (str):
            Path to the nifti file

    Returns:
        original_shape (tuple):
            Original shape of the image
    '''
    # load the header
    header = nib.load(image_file_path).header

    # original shape
    original_shape = header.get_data_shape()

    return original_shape