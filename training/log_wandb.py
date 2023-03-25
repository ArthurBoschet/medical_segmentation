import torch
import wandb
from training.train import train

def log_wandb_run(model, 
                  batch_size,
                  num_classes,
                  resize,
                  dataset,
                  shuffle,
                  train_dataloader, 
                  val_dataloader, 
                  num_epochs=50, 
                  patience=50, 
                  optimizer=None, 
                  criterion=None, 
                  segmentation_ouput=False,
                  ):
    ''' 
    Train a U-Net model
    
    Args:
        model: nn.Module 
            U-Net model to train
        batch_size: int
            Batch size
        num_classes: int
            Number of classes
        TODO: resize, dataset, shuffle
        train_dataloader: torch.utils.data.DataLoader
            Dataloader for training set
        val_dataloader: torch.utils.data.DataLoader
            Dataloader for validation set
        num_epochs: int
            Number of epochs to train the model
        patience: int
            Patience for early stopping
        optimizer: torch.optim.Optimizer
            Optimizer to use for training
        scheduler: torch.optim.lr_scheduler
            Learning rate scheduler
        criterion: torch.nn.modules.loss._Loss
            Loss function to use for training
        segmentation_ouput: bool
            Whether to log segmentation image results
    '''

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model dictionary
    model_dic = {
        "model_name": model.__class__.__name__,
        "num_epochs": num_epochs,
        "patience": patience,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion,
    }

    # data loader dictionary
    dataloader_dic = {
        "batch_size": train_dataloader.batch_size,
        "input_size": resize,
        "dataset": dataset,
        "shuffle": shuffle,
    }

    # initialize wandb
    wandb.init(project="ift6759_project", 
                entity="enzymes", 
                config={
                    "device": device,
                    "model": model_dic,
                    "dataloader": dataloader_dic
                })
    wandb.watch(model, log="all")

    train(model, 
          batch_size,
          num_classes,
          device,
          train_dataloader, 
          val_dataloader, 
          num_epochs=num_epochs, 
          patience=patience, 
          optimizer=optimizer, 
          criterion=criterion, 
          wandb_log=True,
          segmentation_ouput=segmentation_ouput,
          )
    
    #terminate run
    wandb.finish()