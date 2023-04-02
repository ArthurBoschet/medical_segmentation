import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import DiceCELoss
import wandb
from training.train import train
from experiments.make_model import make_model


def train_sweep(
    config=None,
    early_stop_patience=100,
    build_network=make_model("../experiments/configs/unet.json"),
    train_dataloader=None,
    val_dataloader=None
    ):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #setup model / loss function / optimizer /scheduler 
        print("make model")
        model = build_network(config.dropout)

        print("setup optimizer and scheduler")
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, patience=config.patience)
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
              )