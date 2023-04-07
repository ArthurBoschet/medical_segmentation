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
    print("Downloading all weights...")
    for task_name in task_names:
        print(f"- {task_name}...")
        for model_name in model_names:
            print(f"--- Downloading weights for model: {model_name}...")
            if not os.path.exists(os.path.join(weights_dir, task_name, model_name, "v0")):
                # download model weights from wandb
                download_weights_wandb("enzymes", 
                                       task_name, 
                                       model_name, 
                                       "v0", 
                                       weights_dir)
            else:
                print(f"----- Weights for model {model_name} already downloaded, skipping...")