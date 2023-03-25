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
                  ):
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
          )
    
    #terminate run
    wandb.finish()