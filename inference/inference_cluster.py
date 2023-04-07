import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing')
sys.path.append('../models')
sys.path.append('../experiments')

import os
import json
import torch
import argparse

from inference import model_inference
from preprocessing.data_loader import load_test_data
from experiments.make_model import make_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        type=str, default='/home/jaggbow/scratch/clem/dataset',
                        help='Path to the dataset folder')
    parser.add_argument('--output_folder',
                        type=str, default='/home/jaggbow/scratch/clem/inference',
                        help='Name of the output folder')
    parser.add_argument('--batch_size',
                        type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--resize',
                        type=tuple, default=(128, 128, 128),
                        help='Resize the input volume')
    
    # parse arguments
    args = parser.parse_args()
    dataset_path = args.dataset_path
    output_folder = args.output_folder
    batch_size = args.batch_size
    resize = args.resize

    # check that resize is a tuple of 3 dimensions
    assert len(resize) == 3, "Resize must be a tuple of 3 dimensions"

    # loop through all tasks and models
    task_names = ["Task01_BrainTumour",
                  "Task02_Heart", 
                  "Task03_Liver", 
                  "Task04_Hippocampus",
                  "Task05_Prostate", 
                  "Task06_Lung", 
                  "Task07_Pancreas", 
                  "Task08_HepaticVessel", 
                  "Task09_Spleen", 
                  "Task10_Colon"]
    model_configs = {"UNet": "unet.json",
                     "UNetConvSkip": "unet_convskip.json",
                     "SwinUNETR": "swin_unetr.json"}
    print("Performing inference on all tasks and models...")
    for task_name in task_names:
        print(f"- {task_name}...")
        for model_name, model_config in model_configs.items():
            print(f"--- Inference on model: {model_name}...")
            # if not already infered, perform inference on model
            if not os.path.exists(os.path.join(output_folder, task_name, model_name)):
                weights_dir = os.path.join('/home/jaggbow/scratch/clem/weights', task_name, model_name, "v0")
                # if weights directory exists, load weights and perform inference
                if os.path.exists(weights_dir):
                    # load dataset.json file
                    with open(os.path.join(dataset_path, task_name, "dataset.json"), "r") as f:
                        dataset_json = json.load(f)
                    num_classes = len(dataset_json["labels"])
                    num_channels = len(dataset_json["modality"])

                    # load weights into model
                    print("----- Loading weights into model...")
                    if torch.cuda.is_available():
                        model = make_model(config=os.path.join("../experiments/configs", model_config), 
                                        input_shape=(num_channels, resize[0], resize[1], resize[2]), 
                                        num_classes=num_classes)
                        if os.path.exists(weights_dir):
                            weights = torch.load(os.path.join(weights_dir, "last_model.pt"))
                            model.load_state_dict(weights)
                            print("----- Weights loaded successfully")
                        else:
                            raise Exception(f"Weights directory does not exist: {weights_dir}")
                    else:
                        raise Exception("No GPU available")

                    # init parameters
                    shuffle = False
                    normalize = True
                    transform = None
                    output_folder_temp = os.path.join(output_folder, task_name, model_name, "v0")
                    task_folder_path = os.path.join(dataset_path, task_name)
                    output_filenames = sorted(os.listdir(os.path.join(task_folder_path, "test")))
                    output_filenames_idx = [filename[-7:-4] for filename in output_filenames]

                    # load test dataloader
                    test_dataloader = load_test_data(task_folder_path,
                                                    batch_size=batch_size,
                                                    shuffle=shuffle,
                                                    normalize=normalize,
                                                    resize=resize,
                                                    transform=transform)

                    # perform inference on model and save output labels
                    model_inference(model,
                                    test_dataloader,
                                    task_folder_path,
                                    task_name,
                                    output_folder_temp,
                                    output_filenames_idx)
                    
                    print(f"----- Done")
                else:
                    print(f"----- Weights directory does not exist: {weights_dir}")
            else:
                print(f"----- Inference already performed on model: {model_name}")
