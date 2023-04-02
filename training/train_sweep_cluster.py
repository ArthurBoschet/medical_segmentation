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
import optuna

from monai.losses import DiceCELoss

from utils.data_utils import convert_niigz_to_numpy, prepare_dataset_for_training_local
from preprocessing.data_loader import load_data
from experiments.make_model import make_model
from train import train


def objective(trial):
    '''
    Objective function for Optuna optimization
    
    Args:
        trial (optuna.trial.Trial): 
            Optuna trial object
    '''
    # get hyperparameters
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)

    # model config name
    model_config_name = model_config.split("/")[-1].split(".")[0]

    # make model
    model = make_model(model_config, input_shape=input_shape, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=scheduler_patience)
    criterion = DiceCELoss()

    # model dictionary
    model_dic = {
        "model_name": model.__class__.__name__,
        "num_epochs": num_epochs,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion,
        "lr": lr,
        "weight_decay": weight_decay,
        "lr_factor": factor,
        "scheduler_patience": scheduler_patience,
    }

    # data loader dictionary
    dataloader_dic = {
        "batch_size": batch_size,
        "num_classes": num_classes,
        "input_size": train_dataloader.dataset.input_size,
        "dataset": train_dataloader.dataset.dataset_task,
        "shuffle": train_dataloader.dataset.shuffle,
        "normalize": train_dataloader.dataset.normalize,
        "transform": train_dataloader.dataset.transform
    }

    # initialize wandb run
    wandb.init(
        project=project_name,
        entity="enzymes",
        mode="offline",
        dir="/home/jaggbow/scratch/clem/logs/sweep",
        config={
            "device": device,
            "model": model_dic,
            "dataloader": dataloader_dic
        },
        name=f"{model_config_name}_run_{trial.number}",
    )
    wandb.watch(model, log="all")

    # train model
    best_val_dice = train(
        model=model, 
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        batch_size=batch_size,
        num_classes=num_classes,
        num_epochs=num_epochs, 
        patience=num_epochs, 
        optimizer=optimizer, 
        scheduler=scheduler,
        criterion=criterion, 
        wandb_log=True,
        segmentation_ouput=False,
        artifact_log=False,
    )

    #terminate run
    wandb.finish()

    # return best validation dice
    return best_val_dice["max_val_dice_1"]


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
    parser.add_argument('--project_name',
                        type=str, default='unet_sweep',
                        help='Name of the wandb project')
    parser.add_argument('--num_epochs',
                        type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--num_trials', 
                        type=int, default=20,
                        help='Number of trials')
    
    # parse arguments
    args = parser.parse_args()
    model_config = args.model_config
    dataset_path = args.dataset_path
    task_name = args.task_name
    project_name = args.project_name
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
    scheduler_patience = model_config_json["training"]["patience"]
    factor = model_config_json["training"]["factor"]

    # create and run optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_trials)

    # print best parameters
    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best parameters:", study.best_params)
    