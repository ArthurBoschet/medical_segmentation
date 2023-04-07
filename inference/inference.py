import os
import datetime
import numpy as np
import torch

from tqdm import tqdm

from utils.data_utils import save_nifti


def model_inference(model,
                    test_dataloader,
                    dataset_dir,
                    dataset_name,
                    output_folder,
                    output_filenames_idx):
    '''
    Inference on the test set

    Args:
        model: nn.Module
            Model to use for inference
        test_dataloader: torch.utils.data.DataLoader
            Dataloader for the test set
        dataset_dir: str
            Path to the dataset folder
        dataset_name: str
            Name of the dataset
        output_folder: str
            Path to the folder where to save the output
        output_filenames_idx: list(int)
            List of indices of the output filenames index
    '''
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y_%m_%d-%H_%M_%S")

    task_name_dic = {
        "Task01_BrainTumour": "BRATS",
        "Task02_Heart": "la",
        "Task03_Liver": "liver",
        "Task04_Hippocampus": "hippocampus",
        "Task05_Prostate": "prostate",
        "Task06_Lung": "lung",
        "Task07_Pancreas": "pancreas",
        "Task08_HepaticVessel": "hepaticvessel",
        "Task09_Spleen": "spleen",
        "Task10_Colon": "colon",
    }

    if not os.path.exists(os.path.join(output_folder, timestamp_str)):
        os.makedirs(os.path.join(output_folder, timestamp_str))
    output_folder = os.path.join(output_folder, timestamp_str)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_folder_path = os.path.join(dataset_dir, "test")
    filenames = sorted(os.listdir(test_folder_path))

    print("------- Inference on test set...")

    model.eval()
    with torch.no_grad():
        for i, (input, idx) in tqdm(enumerate(zip(test_dataloader, output_filenames_idx))):
            resize = np.load(os.path.join(os.path.join(test_folder_path, filenames[i]))).shape
            if model.__class__.__name__ != "SwinUNETR":
                output = model.predict(input.to(device)).float()
            else:
                output = model(input.to(device))
                output = torch.argmax(output, 1, keepdim=True).float()
            output = torch.nn.functional.interpolate(output, size=tuple((resize[1], resize[2], resize[3])), mode='trilinear')
            output = torch.round(output)
            label = output[0][0].cpu().numpy().astype(np.int8)
            label = np.transpose(label, (1, 2, 0))
            save_nifti(label, affine=np.eye(4), filename=os.path.join(output_folder, f"{task_name_dic[dataset_name]}_{idx}.nii.gz"))