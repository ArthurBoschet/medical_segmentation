import torch
import wandb

from monai.losses import DiceCELoss

from training.train import train
from experiments.make_model import make_model


def train_sweep(
    config=None,
    model_config="../experiments/configs/unet.json",
    early_stop_patience=100,
    train_dataloader=None,
    val_dataloader=None
    ):
    '''
    Train a model with the given config and log the results to wandb.
    
    Args:
        config (dict): 
            A dictionary containing the hyperparameters to use for training.
            If None, the default values will be used.
        model_config (str):
            Path to the model config json file.
        early_stop_patience (int):
            Number of epochs to wait before early stopping.
        train_dataloader (torch.utils.data.DataLoader):
            Dataloader for training set.
        val_dataloader (torch.utils.data.DataLoader):
            Dataloader for validation set.
    '''
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #setup model / loss function / optimizer /scheduler 
        print("make model")
        model = make_model(model_config, input_shape=config.input_shape, num_classes=config.num_classes)

        print("setup optimizer and scheduler")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, patience=config.patience)
        criterion = DiceCELoss()

        print("train")
        train(model, 
              config.batch_size,
              config.num_classes,
              device,
              train_dataloader, 
              val_dataloader, 
              num_epochs=config.epochs, 
              patience=early_stop_patience, 
              optimizer=optimizer, 
              scheduler=scheduler,
              criterion=criterion, 
              wandb_log=True,
              segmentation_ouput=False,
              artifact_log=False,
              )