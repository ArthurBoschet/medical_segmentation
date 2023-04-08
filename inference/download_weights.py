import sys
sys.path.append('../')
sys.path.append('../utils')

import os

from utils.wandb_api import download_weights_wandb


if __name__ == "__main__":
    # loop over all tasks to download all models
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
    model_names = ["UNet", 
                   "UNetConvSkip", 
                   "SwinUNETR"]
    weights_dir = f'/home/jaggbow/scratch/clem/weights'
    version = "v0"
    print("Downloading all weights...")
    for model_name in model_names:
        print(f"- {model_name}...")
        for task_name in task_names:
            print(f"--- Downloading weights for model: {task_name}...")
            if not os.path.exists(os.path.join(weights_dir, model_name, task_name, version)):
                # download model weights from wandb
                download_weights_wandb("enzymes", 
                                       task_name, 
                                       model_name, 
                                       version, 
                                       weights_dir)
            else:
                print(f"----- Weights for model {model_name} already downloaded, skipping...")