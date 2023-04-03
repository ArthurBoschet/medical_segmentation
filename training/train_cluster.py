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

from monai.losses import DiceCELoss

from utils.data_utils import convert_niigz_to_numpy, prepare_dataset_for_training_local
from preprocessing.data_loader import load_data
from experiments.make_model import make_model
from log_wandb import log_wandb_run
from log_wandb_kfold import log_wandb_run_kfold


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
                        type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr',
                        type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--k_folds',
                        type=int, default=0,
                        help='Number of k-folds to use for training (0 for no k-folds)')
    
    # parse arguments
    args = parser.parse_args()
    model_config = args.model_config
    dataset_path = args.dataset_path
    task_name = args.task_name
    num_epochs = args.num_epochs
    lr = args.lr
    k_folds = args.k_folds

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

    # check if we are training on k-folds
    if k_folds==0:
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

        # instantiate model
        model = make_model(config=model_config, input_shape=input_shape, num_classes=num_classes)

        # init parameters
        weight_decay = model_config_json["training"]["weight_decay"]
        patience = model_config_json["training"]["patience"]
        factor = model_config_json["training"]["factor"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = DiceCELoss(to_onehot_y=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        run_name = f"{model.__class__.__name__}_{task_name}"

        # train model
        log_wandb_run(model, 
                      train_dataloader, 
                      val_dataloader, 
                      batch_size=batch_size,
                      num_classes=num_classes, 
                      num_epochs=num_epochs, 
                      patience=100, 
                      optimizer=optimizer, 
                      criterion=criterion, 
                      scheduler=scheduler,
                      segmentation_ouput=True,
                      run_name=run_name,
                      offline=True,
                      wandb_dir="/home/jaggbow/scratch/clem/logs")
    else:
        # train on k-folds and log results
        log_wandb_run_kfold(model_config,
                            task_folder_path,
                            lr=lr,
                            k_folds=k_folds,
                            batch_size=batch_size,
                            num_classes=num_classes,
                            num_epochs=num_epochs,
                            patience=num_epochs,
                            segmentation_ouput=True,
                            offline=True,
                            wandb_dir="/home/jaggbow/scratch/clem/logs")
