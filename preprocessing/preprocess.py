import os
import shutil

from nnUNet.nnunet.experiment_planning.nnUNet_convert_decathlon_task import main as convert_decathlon_task
from nnUNet.nnunet.experiment_planning.nnUNet_plan_and_preprocess import main as preprocess_task
from utils.data_utils import convert_to_numpy


def preprocess_data(task_folder_path):
    '''
    Convert the data into the nnUNet format and preprocess it
    Refactor the structure of the data folder
    
    Args:
        task_folder_path: str 
            Path to the task folder

    The task_folder_path structure should correspond to:

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
    
    Output folder structure will be:

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
    │   ├── stage_1 (if apllicable)
    │   │   ├── imagesTr
    │   │   │   ├── image_000.npy
    │   │   │   ├── ...
    │   │   ├── labelsTr
    │   │   │   ├── label_000.npy
    │   │   │   ├── ...
    ├── dataset.json
    '''
    # verify the structure of the input task folder
    assert os.path.exists(os.path.join(task_folder_path, 'imagesTr')), "imagesTr folder does not exist"
    assert os.path.exists(os.path.join(task_folder_path, 'labelsTr')), "labelsTr folder does not exist"
    assert os.path.exists(os.path.join(task_folder_path, 'imagesTs')), "imagesTs folder does not exist"
    assert os.path.exists(os.path.join(task_folder_path, 'dataset.json')), "dataset.json file does not exist"
 
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

    # rename files
    for dataset_path in [os.path.join(task_folder_path, 'raw'), os.path.join(task_folder_path, 'preprocessed', 'stage_0'), os.path.join(task_folder_path, 'preprocessed', 'stage_1')]:
        if os.path.exists(dataset_path):
            for subdir in os.listdir(dataset_path):
                if os.path.isdir(os.path.join(dataset_path, subdir)):
                    for filename in os.listdir(os.path.join(dataset_path, subdir)):
                        if filename.endswith('.nii.gz'):
                            if subdir == 'imagesTr' or subdir == 'imagesTs':
                                start_idx = filename.find("_")+1
                                stop_idx = filename.find("_", start_idx, len(filename))
                                ext_idx = filename.find(".nii.gz")
                                new_filename = filename[:start_idx] + filename[start_idx:stop_idx].zfill(3) + filename[ext_idx:]
                            elif subdir == 'labelsTr':
                                start_idx = filename.find("_")+1
                                stop_idx = filename.find(".nii.gz")
                                new_filename = filename[:start_idx] + filename[start_idx:stop_idx].zfill(3) + filename[stop_idx:]
                        elif filename.endswith('.npz'):
                            start_idx = filename.find("_")+1
                            stop_idx = filename.find(".npz")
                            new_filename = filename[:start_idx] + filename[start_idx:stop_idx].zfill(3) + filename[stop_idx:]
                        os.rename(os.path.join(dataset_path, subdir, filename), os.path.join(dataset_path, subdir, new_filename))

    # remove useless folders
    shutil.rmtree(os.path.join(task_folder_path, 'preprocessed_data'))
    shutil.rmtree(os.path.join(task_folder_path, 'imagesTr'))
    shutil.rmtree(os.path.join(task_folder_path, 'imagesTs'))
    shutil.rmtree(os.path.join(task_folder_path, 'labelsTr'))

    # remove useless files
    os.remove(os.path.join(task_folder_path, 'raw', 'dataset.json'))

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
