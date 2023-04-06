import os
import shutil

from tqdm import tqdm


def prepare_dataset_for_training(dataset_folder_path, output_dataset_path=None):
    '''
    Prepare the dataset for training by splitting it into training and validation sets for the dataloaders

    Args:
        dataset_folder_path: str
            Path to the dataset folder
        output_dataset_path: str
            Path to the output dataset folder that will be created and used for training
    '''
    # check if output_dataset_path exists (which mean we want to use a google colab notebook to train the model)
    # otherwise we do not need to copy the dataset to the virtual machine local disk as we will have access to the dataset folder in the compute canada scratch folder
    if output_dataset_path is not None:
        # copy dataset from drive to virtual machine local disk
        shutil.copytree(dataset_folder_path, output_dataset_path)
    else:
        # use the dataset folder in the compute canada scratch folder as output dataset path
        output_dataset_path = dataset_folder_path

    # paths
    train_images_files = sorted(os.listdir(os.path.join(output_dataset_path, "imagesTr")))
    train_labels_files = sorted(os.listdir(os.path.join(output_dataset_path, "labelsTr")))
    test_images_files = sorted(os.listdir(os.path.join(dataset_folder_path, "imagesTs")))

    # create train, val and test folders
    os.makedirs(os.path.join(output_dataset_path, "train_val"))
    os.makedirs(os.path.join(output_dataset_path, "test"))

    # copy training images and labels to train_val folders
    for tr_im_file, tr_label_file in tqdm(zip(train_images_files, train_labels_files)):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTr", tr_im_file), os.path.join(output_dataset_path, "train_val", tr_im_file))
        shutil.copyfile(os.path.join(dataset_folder_path, "labelsTr", tr_label_file), os.path.join(output_dataset_path, "train_val", tr_label_file))
    # copy test images to test folder
    for test_im_file in tqdm(test_images_files):
        shutil.copyfile(os.path.join(dataset_folder_path, "imagesTs", test_im_file), os.path.join(output_dataset_path, "test", test_im_file))
    
    # remove useless folders
    shutil.rmtree(os.path.join(output_dataset_path, "imagesTr"))
    shutil.rmtree(os.path.join(output_dataset_path, "labelsTr"))
    shutil.rmtree(os.path.join(output_dataset_path, "imagesTs"))
