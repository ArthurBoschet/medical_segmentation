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

def check_for_padding(input_folder):
    '''
    Check if the images and labels need to be padded
    Add padding if necessary

    Args:
        input_folder: str 
            Path to the folder containing the images and labels
    '''
    for subdir in ['stage_0', 'stage_1']:
        if os.path.exists(os.path.join(input_folder, subdir)):
            max_width = 0
            max_height = 0
            padding_needed = False
            for file in tqdm(os.listdir(os.path.join(input_folder, subdir, "imagesTr"))):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(input_folder, subdir, "imagesTr", file))
                    if data.shape[1] != data.shape[2]:
                        padding_needed = True
                    if data.shape[2] > max_width:
                        max_width = data.shape[2]
                    if data.shape[1] > max_height:
                        max_height = data.shape[1]
            if padding_needed:
                for subsubdir in ['imagesTr', 'labelsTr']:
                    dir = os.path.join(input_folder, subdir, subsubdir)
                    out_dir = os.path.join(f'{input_folder}_pad', subdir, subsubdir)
                    os.makedirs(out_dir, exist_ok=True)
                    for file in tqdm(os.listdir(dir)):
                        if file.endswith('.npy'):
                            data = np.load(os.path.join(dir, file))
                            data = np.pad(data, ((0, 0), (0, max_height-data.shape[1]), (0, max_width-data.shape[2])), 'constant')
                            np.save(os.path.join(out_dir, file), data)

def get_resize_shape(input_folder, factor=2):
    '''
    Get the shape to resize the images and labels from the dataset to
    
    Args:
        input_folder: str 
            Path to the folder containing the images and labels
        factor: int
            Factor to resize the images and labels by
    '''
    width_height_mean = 0
    depth_mean = 0
    for subdir in ['imagesTr', 'labelsTr']:
        dir = os.path.join(input_folder, subdir)
        if os.path.isdir(dir):
            for file in tqdm(os.listdir(dir)):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(dir, file))
                    width_height_mean += data.shape[1] + data.shape[2]
                    depth_mean += data.shape[0]
    width_height_mean /= len(os.listdir(dir)) * 2
    depth_mean /= len(os.listdir(dir))
    return (int(depth_mean/factor), int(width_height_mean/factor), int(width_height_mean/factor))
