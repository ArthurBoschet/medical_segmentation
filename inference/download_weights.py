import sys
sys.path.append('../')
sys.path.append('../utils')

import argparse

from utils.wandb_api import download_weights_wandb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str, required=True,
                        help='Name of the model artifact on wandb')
    parser.add_argument('--task_name', required=True,
                        type=str, default='Task02_Heart',
                        help='Name of the task')
    parser.add_argument('--version',
                        type=str, required=True,
                        help='Version of the model on wandb')
    
    # parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    task_name = args.task_name
    version = args.version

    # download model weights from wandb
    download_weights_wandb("enzymes", 
                           task_name, 
                           model_name, 
                           version, 
                           '/home/jaggbow/scratch/clem/weights')