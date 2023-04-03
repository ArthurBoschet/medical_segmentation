import sys
sys.path.append('../')
sys.path.append('../preprocessing')
sys.path.append('../experiments')

import torch
import json

from monai.losses import DiceCELoss

from log_wandb import log_wandb_run
from preprocessing.data_loader import load_data_kfold
from experiments.make_model import make_model


def log_wandb_run_kfold(model_config, 
                        data_folder_path,
                        lr=1e-3,
                        k_folds=5,
                        batch_size=2,
                        num_classes=2, 
                        num_epochs=50, 
                        patience=50, 
                        segmentation_ouput=False,
                        run_name=None,
                        offline=False,
                        wandb_dir=None
                        ):
    ''' 
    Train a U-Net model
    
    Args:
        model: nn.Module 
            U-Net model to train
        data_folder_path: str
            Path to folder containing data
        k_folds: int
            Number of k-folds to use
        batch_size: int
            Batch size
        num_classes: int
            Number of classes
        num_epochs: int
            Number of epochs to train the model
        patience: int
            Patience for early stopping
        optimizer: torch.optim.Optimizer
            Optimizer to use for training
        criterion: torch.nn.modules.loss._Loss
            Loss function to use for training
        scheduler: torch.optim.lr_scheduler
            Learning rate scheduler
        segmentation_ouput: bool
            Whether to log segmentation image results
        run_name: str
            Name of the run to log to wandb
        offline: bool
            Whether to log to wandb offline
        wandb_dir: str
            Directory to save wandb logs
    '''
    # read the model config json file
    with open(model_config, 'r') as f:
        model_config_json = json.load(f)
    
    # kfold cross validation loop
    for fold in range(k_folds):

        print(f"Fold {fold + 1}/{k_folds}")

        # get the corresponding data loaders
        train_dataloader_kfold, val_dataloader_kfold = load_data_kfold(data_folder_path,
                                                                       k_folds=k_folds,
                                                                       fold=fold,
                                                                       batch_size=batch_size,
                                                                       num_classes=num_classes,
                                                                       shuffle=True,
                                                                       normalize=True,
                                                                       resize=tuple(model_config_json['training']['resize']),
                                                                       transform=None)
        
        # init params 
        weight_decay = model_config_json['training']['weight_decay']
        lr_factor = model_config_json['training']['factor']
        scheduler_patience = model_config_json['training']['patience']

        # init model
        input_shape = (1, train_dataloader_kfold.dataset.input_size[0], train_dataloader_kfold.dataset.input_size[1], train_dataloader_kfold.dataset.input_size[2])
        model = make_model(model_config, input_shape=input_shape, num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = DiceCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=scheduler_patience)

        # set run name
        model_name = model_config.split('/')[-1].split('.')[0]
        run_name = f"{model_name}_fold_{fold + 1}"

        # train and log to wandb
        _ = log_wandb_run(model, 
                          train_dataloader_kfold, 
                          val_dataloader_kfold, 
                          batch_size,
                          num_classes,
                          num_epochs=num_epochs, 
                          patience=patience, 
                          optimizer=optimizer, 
                          criterion=criterion, 
                          scheduler=scheduler,
                          segmentation_ouput=segmentation_ouput if fold == 0 else False,
                          run_name=run_name,
                          offline=offline,
                          wandb_dir=wandb_dir,
                          )