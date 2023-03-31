import os
import datetime
import numpy as np
import nibabel as nib
import torch
import wandb

from utils.data_utils import save_nifti, get_original_shape


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
    test_folder_path = os.path.join(dataset_dir, dataset_name, "test")
    filenames = sorted(os.listdir(test_folder_path))

    model.eval()
    with torch.no_grad():
        for i, (input, idx) in enumerate(zip(test_dataloader, output_filenames_idx)):
            resize = np.load(os.path.join(os.path.join(test_folder_path, filenames[i]))).shape
            output = model.predict(input.to(device)).float()
            output = torch.nn.functional.interpolate(output, size=(resize), mode='trilinear')
            output = torch.round(output)
            label = output[0][0].cpu().numpy().astype(np.int8)
            label = np.transpose(label, (1, 2, 0))
            save_nifti(label, affine=np.eye(4), filename=os.path.join(output_folder, f"{task_name_dic[dataset_name]}_{idx}.nii.gz"))


def download_model_wandb(network, username, project_name, artifact_name, artifact_version):
    '''
    Download model weights from wandb

    Args:
        network: nn.Module
            PyTorch model to load the weights into
        username: str
            Username of the wandb account
        project_name: str
            Name of the wandb project
        artifact_name: str
            Name of the wandb artifact
        artifact_version: str
            Version of the wandb artifact

    Returns:
        network: nn.Module
            PyTorch model with the weights loaded
    '''
    # set up the api instance
    artifact_path = os.path.join(username, project_name, f"{artifact_name}:{artifact_version}")

    # set up the weights artifact
    api = wandb.Api()
    artifact = api.artifact(artifact_path)

    # download the artifact
    weights_path = artifact.download()

    # load weights into model
    weights = torch.load(os.path.join(weights_path, "best_model.pt"))
    network.load_state_dict(weights)

    print(f"Model weights downloaded to {weights_path} and loaded into PyTorch model correctly")

    return network
