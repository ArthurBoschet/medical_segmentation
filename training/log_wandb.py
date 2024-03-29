import torch
import wandb
from training.train import train
from training.train import train_without_validation


def log_wandb_run(model, 
                  train_dataloader, 
                  val_dataloader,
                  batch_size=2,
                  num_classes=2, 
                  num_epochs=50, 
                  patience=50, 
                  optimizer=None, 
                  criterion=None, 
                  scheduler=None,
                  segmentation_ouput=False,
                  run_name=None,
                  offline=False,
                  wandb_dir=None,
                  train_only=False
                  ):
    ''' 
    Train a U-Net model
    
    Args:
        model: nn.Module 
            U-Net model to train
        train_dataloader: torch.utils.data.DataLoader
            Dataloader for training set
        val_dataloader: torch.utils.data.DataLoader
            Dataloader for validation set
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
        train_only: bool
            Whether to train on train set only
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
        "lr": optimizer.param_groups[0]['lr'],
    }

    # data loader dictionary
    dataloader_dic = {
        "batch_size": batch_size,
        "num_classes": num_classes,
        "num_train_samples": len(train_dataloader.dataset),
        "input_size": train_dataloader.dataset.input_size,
        "dataset": train_dataloader.dataset.dataset_task,
        "shuffle": train_dataloader.dataset.shuffle,
        "normalize": train_dataloader.dataset.normalize,
        "transform": train_dataloader.dataset.transform
    }

    # initialize wandb
    wandb.init(project=dataloader_dic["dataset"], 
                entity="enzymes", 
                name=run_name,
                config={
                    "device": device,
                    "model": model_dic,
                    "dataloader": dataloader_dic
                },
                mode="offline" if offline else "online",
                dir=wandb_dir)
    wandb.watch(model, log="all")

    if train_only:
        print("Training on train set only")
        _ = train_without_validation(model,
                                     train_dataloader,
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     num_epochs=num_epochs,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     wandb_log=True,
                                     segmentation_ouput=segmentation_ouput,
                                     artifact_log=True,
                                     )
    else:
        _ = train(model, 
                train_dataloader, 
                val_dataloader, 
                batch_size,
                num_classes,
                num_epochs=num_epochs, 
                patience=patience, 
                optimizer=optimizer, 
                criterion=criterion, 
                scheduler=scheduler,
                wandb_log=True,
                segmentation_ouput=segmentation_ouput,
                artifact_log=False,
                )
    
    #terminate run
    wandb.finish()