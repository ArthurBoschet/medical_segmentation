import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing')
sys.path.append('../models')
sys.path.append('../experiments')

import os
import json
import torch
import wandb
import argparse

from utils.data_utils import convert_niigz_to_numpy, prepare_dataset_for_training_local
from preprocessing.data_loader import load_data
from wandb_sweep import train_sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config',
                        type=str, default='../experiments/configs/unet.json',
                        help='Path to the model config json file')
    parser.add_argument('--dataset_path',
                        type=str, default='/home/jaggbow/scratch/clem/dataset', 
                        help='Path to the dataset folder')
    parser.add_argument('--task_name',
                        type=str, default='Task02_Heart',
                        help='Name of the task')
    parser.add_argument('--num_epochs',
                        type=int, default=60,
                        help='Number of epochs')
    parser.add_argument('--num_trials', 
                        type=int, default=10,
                        help='Number of trials')
    
    # parse arguments
    args = parser.parse_args()
    model_config = args.model_config
    dataset_path = args.dataset_path
    task_name = args.task_name
    num_epochs = args.num_epochs
    num_trials = args.num_trials

    # open json file (model config)
    with open(model_config, "r") as f:
        model_config_json = json.load(f)
    
    # get batch size and resize from model_config_json
    batch_size = model_config_json["training"]["batch_size"]
    resize = model_config_json["training"]["resize"]

    # path to the task folder
    task_folder_path = os.path.join(dataset_path, task_name)

    # load dataset.json file
    with open(os.path.join(task_folder_path, "dataset.json"), "r") as f:
        dataset_json = json.load(f)

    # check if the task folder is ready for training
    if not os.path.exists(os.path.join(task_folder_path, "train_val")) and not os.path.exists(os.path.join(task_folder_path, "test")):
        assert os.path.exists(os.path.join(task_folder_path, "imagesTr")), "imagesTr folder does not exist"
        assert os.path.exists(os.path.join(task_folder_path, "labelsTr")), "labelsTr folder does not exist"
        assert os.path.exists(os.path.join(task_folder_path, "imagesTs")), "imagesTs folder does not exist"
        assert os.path.exists(os.path.join(task_folder_path, "dataset.json")), "dataset.json file does not exist"
        if sorted(os.listdir(os.path.join(task_folder_path, "imagesTr")))[0].endswith(".nii.gz"):
            print("Converting nii.gz to numpy...")
            convert_niigz_to_numpy(task_folder_path)
            print("Done")
        elif sorted(os.listdir(os.path.join(task_folder_path, "imagesTr")))[0].endswith(".npy"):
            assert len(os.listdir(os.path.join(task_folder_path, "imagesTr")))==dataset_json["numTraining"], "Number of training images does not match dataset.json"
            assert len(os.listdir(os.path.join(task_folder_path, "labelsTr")))==dataset_json["numTraining"], "Number of training labels does not match dataset.json"
            assert len(os.listdir(os.path.join(task_folder_path, "imagesTs")))==dataset_json["numTest"], "Number of test images does not match dataset.json"
            print("Numpy files already exist")
        print("Preparing dataset for training...")
        prepare_dataset_for_training_local(task_folder_path)
        print("Done")
    else:
        assert len(os.listdir(os.path.join(task_folder_path, "train_val")))==dataset_json["numTraining"]*2, "Number of training images and labels does not match dataset.json"
        assert len(os.listdir(os.path.join(task_folder_path, "test")))==dataset_json["numTest"], "Number of test images does not match dataset.json"
        print("Task folder is already ready for training!")

    # init parameters
    num_classes = len(dataset_json["labels"])
    shuffle = True
    normalize = True
    transform = None

    # load dataloaders
    train_dataloader, val_dataloader = load_data(task_folder_path, 
                                                 batch_size=batch_size, 
                                                 num_classes=num_classes, 
                                                 shuffle=shuffle,
                                                 normalize=normalize,
                                                 resize=resize,
                                                 transform=transform)

    # shapes
    input_example = train_dataloader.dataset[0][0].unsqueeze(0)
    input_shape = tuple(list(input_example[0].shape))

    # setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is: {device}")

    # init constant parameters
    weight_decay = model_config_json["training"]["weight_decay"]
    patience = model_config_json["training"]["patience"]
    factor = model_config_json["training"]["factor"]

    # setup the sweep 
    sweep_config = {
        'method': 'bayes'
    }

    # the goal is to maximize the validation dice score of the foreground
    metric = {
        'name': 'max_val_dice_1',
        'goal': 'maximize'   
    }

    early_terminate = {
        'type': 'hyperband',
        'min_iter': 30
    }

    sweep_config['early_terminate'] = early_terminate

    sweep_config['metric'] = metric

    # hyper parameter space
    parameters_dict = {
        # constant values 
        'batch_size' : {
            'value': 2
        },
        'epochs': {
            'value': num_epochs
        },
        'num_classes':{
            'value': num_classes
        },
        'input_shape': {
            'value': input_shape
        },
        'weight_decay': {
            'value': weight_decay
        },
        'patience': {
            'value': patience
        },
        'lr_factor': {
            'value': factor
        },

        # optimizer variables
        'lr': {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-1
        },
    }
    sweep_config['parameters'] = parameters_dict
    
    sweep_trainer = lambda: train_sweep(
        config=None,
        model_config=model_config,
        early_stop_patience=100,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader
    )

    # run the sweep
    sweep_id = wandb.sweep(sweep_config, project="unet_colon_sweep", entity='enzymes')
    wandb.agent(sweep_id, sweep_trainer, count=num_trials)