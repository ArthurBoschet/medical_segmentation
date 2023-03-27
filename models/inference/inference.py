import os
import datetime
import numpy as np
import nibabel as nib
import torch

from utils.data_utils import reconstruct_affine_matrix, save_nifti


def model_inference(model,
                    test_dataloader,
                    dataset_name,
                    original_dataset_dir,
                    output_folder,
                    output_filenames_idx):
    '''
    Inference on the test set

    Args:
        model: nn.Module
            Model to use for inference
        test_dataloader: torch.utils.data.DataLoader
            Dataloader for the test set
        dataset_name: str
            Name of the dataset
        original_dataset_dir: str
            Path to the original dataset
        output_folder: str
            Path to the folder where to save the output
        output_filenames_idx: list(int)
            List of indices of the output filenames index
    '''
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d-%H_%M_%S")

    task_name_dic = {
        "Task02_Heart": "la",
        "Task07_Pancreas": "pancreas",
        "Task09_Spleen": "spleen",
        "Task10_Colon": "colon",
    }

    if not os.path.exists(os.path.join(output_folder, timestamp_str)):
        os.makedirs(os.path.join(output_folder, timestamp_str))
    output_folder = os.path.join(output_folder, timestamp_str)

    header_path = os.path.join(original_dataset_dir, dataset_name, "labelsTr")
    header_filenames = sorted(os.listdir(header_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        for i, (input, idx) in enumerate(zip(test_dataloader, output_filenames_idx)):
            output = model.predict(input.to(device))
            label = output[0][0].cpu().numpy().astype(np.int16)
            header_file_path = os.path.join(header_path, header_filenames[i])
            affine_matrix = reconstruct_affine_matrix(header_file_path)
            save_nifti(label, affine_matrix, os.path.join(output_folder, f"{task_name_dic[dataset_name]}_{idx}.nii.gz"))
