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

def get_resize_shape(input_folder, factor=2):
    '''
    Get the shape to resize the images and labels from the dataset to
    
    Args:
        input_folder: str 
            Path to the folder containing the images and labels
        factor: int
            Factor to resize the images and labels by
    '''
    width_mean = 0
    height_mean = 0
    depth_mean = 0
    for subdir in ["train", "val"]:
        dir = os.path.join(input_folder, subdir)
        if os.path.isdir(dir):
            for file in tqdm(os.listdir(dir)):
                if file.endswith('.npy'):
                    if file.startswith('image'):
                        data = np.load(os.path.join(dir, file))
                        depth_mean += data.shape[0]
                        height_mean += data.shape[1]
                        width_mean += data.shape[2]
    depth_mean /= (len(os.listdir(os.path.join(input_folder, "train")))+len(os.listdir(input_folder, "val")))//2
    height_mean /= (len(os.listdir(os.path.join(input_folder, "train")))+len(os.listdir(input_folder, "val")))//2
    width_mean /= (len(os.listdir(os.path.join(input_folder, "train")))+len(os.listdir(input_folder, "val")))//2
    depth_mean /= factor
    height_mean /= factor
    width_mean /= factor
    depth_mean = int(np.ceil(depth_mean/factor) * factor)
    height_mean = int(np.ceil(height_mean/factor) * factor)
    width_mean = int(np.ceil(width_mean/factor) * factor)
    return (depth_mean, height_mean, width_mean)

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

    # copy validation images and labels to val folder
    for val_im_file, val_label_file in zip(val_images_files, val_labels_files):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTr", val_im_file), os.path.join(output_dataset_path, "val", val_im_file))
        shutil.copyfile(os.path.join(dataset_folder_path, "labelsTr", val_label_file), os.path.join(output_dataset_path, "val", val_label_file))
    # copy training images and labels to train folder
    for tr_im_file, tr_label_file in zip(train_images_files, train_labels_files):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTr", tr_im_file), os.path.join(output_dataset_path, "train", tr_im_file))
        shutil.copyfile(os.path.join(dataset_folder_path, "labelsTr", tr_label_file), os.path.join(output_dataset_path, "train", tr_label_file))

    # renames train files
    for i, (im_tr, la_tr) in enumerate(zip(sorted(train_images_files), sorted(train_labels_files))):
        os.rename(os.path.join(output_dataset_path, "train", im_tr), os.path.join(output_dataset_path, "train", f"image_{str(i).zfill(3)}.npy"))
        os.rename(os.path.join(output_dataset_path, "train", la_tr), os.path.join(output_dataset_path, "train", f"label_{str(i).zfill(3)}.npy"))
    # renames val files
    for i, (im_val, la_val) in enumerate(zip(sorted(val_images_files), sorted(val_labels_files))):
        os.rename(os.path.join(output_dataset_path, "val", im_val), os.path.join(output_dataset_path, "val", f"image_{str(i).zfill(3)}.npy"))
        os.rename(os.path.join(output_dataset_path, "val", la_val), os.path.join(output_dataset_path, "val", f"label_{str(i).zfill(3)}.npy"))

    # remove useless folders
    shutil.rmtree(os.path.join(output_dataset_path, "imagesTr"))
    shutil.rmtree(os.path.join(output_dataset_path, "labelsTr"))
