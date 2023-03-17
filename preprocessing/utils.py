import os
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
            np.save(os.path.join(image_dir, f'image_{file[-7:-4]}.npy'), data[0])
            np.save(os.path.join(label_dir, f'label_{file[-7:-4]}.npy'), data[1])
            os.remove(os.path.join(image_dir, file))

def convert_niigz_to_numpy(input_folder):
    '''
    Convert the images and labels from .nii.gz files into .npy (numpy) files
    
    Args:
        input_folder: str
            Path to the folder containing the images and labels
    '''
    for type in ['imagesTr', 'labelsTr', 'imagesTs']:
        dir = os.path.join(input_folder, type)
        for file in tqdm(os.listdir(dir)):
            if file.endswith('.nii.gz'):
                data = nib.load(os.path.join(dir, file)).get_fdata()
                data = np.transpose(data, (2, 0, 1))
                np.save(os.path.join(dir, f'{type[:-3]}_{file[-10:-7]}.npy'), data)
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

def add_padding(input_folder):
    '''
    Add padding to the images and labels to make them all the same size

    Args:
        input_folder: str 
            Path to the folder containing the images and labels
    '''
    max_width = 0
    dir = os.path.join(input_folder, 'imagesTr')
    for file in tqdm(os.listdir(dir)):
        if file.endswith('.npy'):
            data = np.load(os.path.join(dir, file))
            if data.shape[2] > max_width:
                max_width = data.shape[2]
    for type in ['imagesTr', 'labelsTr']:
        dir = os.path.join(input_folder, type)
        out_dir = os.path.join(f'{input_folder[:-8]}_pad/{input_folder[-7:]}', type)
        os.makedirs(out_dir, exist_ok=True)
        for file in tqdm(os.listdir(dir)):
            if file.endswith('.npy'):
                data = np.load(os.path.join(dir, file))
                data = np.pad(data, ((0, 0), (0, 0), (0, max_width-data.shape[2])), 'constant')
                np.save(os.path.join(out_dir, file), data)

def get_data(input_folder):
    '''
    Get the images and labels from the input folder
    
    Args:
        input_folder: str 
            Path to the folder containing the images and labels

    Returns:
        images: list(numpy.ndarray)
            List of numpy arrays containing the images
        labels: list(numpy.ndarray)
            List of numpy arrays containing the labels
    '''
    images = []
    labels = []
    for type in ['imagesTr', 'labelsTr']:
        dir = os.path.join(input_folder, type)
        for file in tqdm(sorted(os.listdir(dir))):
            if file.endswith('.npy'):
                data = np.load(os.path.join(dir, file))
                if type == 'imagesTr':
                    images.append(data)
                elif type == 'labelsTr':
                    labels.append(data)
    return images, labels
