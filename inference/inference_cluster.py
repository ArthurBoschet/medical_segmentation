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
    parser.add_argument('--model_config',
                        type=str, required=True,
                        help='Path to the model config json file')
    parser.add_argument('--model_name',
                        type=str, required=True,
                        help='Name of the model artifact on wandb')
    parser.add_argument('--version',
                        type=str, required=True,
                        help='Version of the model on wandb in format v1, v2, etc.')
    parser.add_argument('--task_name', required=True,
                        type=str, default='Task02_Heart',
                        help='Name of the task')
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
    model_config = args.model_config
    model_name = args.model_name
    version = args.version
    dataset_path = args.dataset_path
    task_name = args.task_name
    output_folder = args.output_folder
    batch_size = args.batch_size
    resize = args.resize

    assert os.path.exists(os.path.join(dataset_path, task_name)), "Task folder does not exist"
    assert os.path.exists(os.path.join(dataset_path, task_name, "dataset.json")), "dataset.json file does not exist"
    assert os.path.exists(os.path.join(dataset_path, task_name, "test")), "Test folder does not exist"
    assert len(resize) == 3, "Resize must be a tuple of 3 dimensions"

    # setup weights directory
    weights_dir = os.path.join('/home/jaggbow/scratch/clem/weights', task_name, model_name, version)

    # load dataset.json file
    with open(os.path.join(dataset_path, task_name, "dataset.json"), "r") as f:
        dataset_json = json.load(f)
    num_classes = len(dataset_json["labels"])

    # load weights into model
    print("Loading weights into model")
    if torch.cuda.is_available():
        model = make_model(config=model_config, input_shape=(1, resize[0], resize[1], resize[2]), num_classes=num_classes)
        if os.path.exists(weights_dir):
            weights = torch.load(os.path.join(weights_dir, "best_model.pt"))
            model.load_state_dict(weights)
            print("Weights loaded successfully")
        else:
            raise Exception(f"Weights directory does not exist: {weights_dir}")
    else:
        raise Exception("No GPU available")

    # init parameters
    shuffle = False
    normalize = True
    transform = None
    output_folder = os.path.join(output_folder, task_name, model_name, version)
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
                    output_folder,
                    output_filenames_idx)
