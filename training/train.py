import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Dice, F1Score
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from models.evaluation.metrics import iou_score


def train(model, 
          train_dataloader, 
          val_dataloader,
          batch_size,
          num_classes, 
          num_epochs=50, 
          patience=50, 
          optimizer=None, 
          criterion=None, 
          scheduler=None,
          wandb_log=False,
          segmentation_ouput=False,
          artifact_log=True,
          ):
    ''' 
    Train a model
    
    Args:
        model: nn.Module 
            Model to train
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
        scheduler: torch.optim.lr_scheduler
            Learning rate scheduler
        criterion: torch.nn.modules.loss._Loss
            Loss function to use for training
        wandb_log: bool
            Whether to log to wandb or not
        segmentation_ouput: bool
            Whether to log segmentation image results
        artifact_log: bool
            Whether to log model as artifact

    Returns:
        val_dice_max: dict
            Dictionary of maximum validation dice scores
    '''

    # wandb dir run
    run_dir = ""
    if wandb_log:
        run_dir = wandb.run.dir

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize loss and dice scores
    train_loss = 0
    train_dice = [0 for i in range(num_classes)]
    train_iou = [0 for i in range(num_classes)]
    train_f1_macro = 0
    val_loss = 0
    val_dice = [0 for i in range(num_classes)]
    val_iou = [0 for i in range(num_classes)]
    val_f1_macro = 0

    # initialize lists to store loss and dice scores
    train_loss_list = []
    train_dice_list = [[] for i in range(num_classes)]
    train_iou_list = [[] for i in range(num_classes)]
    train_f1_macro_list = []
    val_loss_list = []
    val_dice_list = [[] for i in range(num_classes)]
    val_iou_list = [[] for i in range(num_classes)]
    val_f1_macro_list = []
    slices_dic = {}

    # initialize training params
    best_val_dice = 0
    best_epoch = 0
    patience_count = 0

    # setup torchmetrics
    dice = Dice(num_classes=1, average='micro').to(device)
    if num_classes>2:
        f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    else:
        f1 = F1Score(task='binary', num_classes=num_classes, average='macro').to(device)

    print("-------------- START TRAINING -------------- ")
    
    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # train
        model.train()
        for j, (images, labels) in tqdm(enumerate(train_dataloader), desc="training", dynamic_ncols=True):

            # move images and labels to device
            images = images.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            if model.__class__.__name__ == "SwinUNETR":
                outputs = model(images)
                predict_softmax = nn.Softmax(dim=1)
                model_predict_proba = predict_softmax(outputs)
            else:
                model_predict_proba = model.predict_proba(images)
            loss = criterion(model_predict_proba, labels)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            train_loss += loss.item()
            train_f1_macro += f1(model_predict_proba.argmax(1, keepdim=True), labels.argmax(1, keepdim=True))
            for i in range(num_classes):
              train_dice[i] += dice(model_predict_proba[:,i:i+1,:,:,:].reshape(-1), labels[:,i:i+1,:,:,:].reshape(-1))
              train_iou[i] += iou_score(model_predict_proba[:,i:i+1,:,:,:].reshape(-1), labels[:,i:i+1,:,:,:].reshape(-1))


        # validation
        model.eval()
        with torch.no_grad():
            for j, (images, labels) in tqdm(enumerate(val_dataloader), desc="validation", dynamic_ncols=True):

                # move images and labels to device
                images = images.to(device)
                labels = labels.to(device)

                # forward
                if model.__class__.__name__ == "SwinUNETR":
                    outputs = model(images)
                    predict_softmax = nn.Softmax(dim=1)
                    model_predict_proba = predict_softmax(outputs)
                else:
                    model_predict_proba = model.predict_proba(images)
                loss = criterion(model_predict_proba, labels)

                # statistics
                val_loss += loss.item()
                val_f1_macro += f1(model_predict_proba.argmax(1, keepdim=True), labels.argmax(1, keepdim=True))
                for i in range(num_classes):
                  val_dice[i] += dice(model_predict_proba[:,i:i+1,:,:,:].reshape(-1), labels[:,i:i+1,:,:,:].reshape(-1))
                  val_iou[i] += iou_score(model_predict_proba[:,i:i+1,:,:,:].reshape(-1), labels[:,i:i+1,:,:,:].reshape(-1))

                #log segmentation output to wandb
                if j == 0 and epoch%5 == 0 and wandb_log and segmentation_ouput:
                    val_im = images[0]
                    val_label = labels[0]
                    if model.__class__.__name__ == "SwinUNETR":
                        output = model(val_im.unsqueeze(0).to(device))
                        output = torch.argmax(output, 1, keepdim=True)
                    else:
                        output = model.predict(val_im.unsqueeze(0).to(device))
                    val_im = val_im.cpu().numpy()[0]
                    val_label = val_label.cpu().numpy()[0]
                    output = output[0].cpu().numpy()[0]
                    for s in range(val_im.shape[1]):
                        fig, ax = plt.subplots(1, 2)
                        for i, (label_type, plot_title) in enumerate(zip([val_label, output], ["Ground Truth", "Prediction"])):
                            ax[i].imshow(val_im[s], cmap="gray")
                            ax[i].imshow(label_type[s], cmap="jet", alpha=0.3)
                            ax[i].set_title(plot_title)
                        slices_dic = slices_dic | {f"slice_{s}": fig}
                        plt.close()

        # calculate average loss and dice scores
        adjust_factor_train = 1 if len(train_dataloader) % 2 == 0 else len(train_dataloader)/(len(train_dataloader)+1)
        adjust_factor_val = 1 if len(val_dataloader) % 2 == 0 else len(val_dataloader)/(len(val_dataloader)+1)

        train_loss = train_loss / len(train_dataloader) * batch_size * adjust_factor_train
        for i in range(num_classes):
          train_dice[i] = train_dice[i] / len(train_dataloader.dataset) * batch_size * adjust_factor_train
          train_iou[i] = train_iou[i] / len(train_dataloader.dataset) * batch_size * adjust_factor_train
          val_dice[i] = val_dice[i] / len(val_dataloader.dataset) * batch_size * adjust_factor_val
          val_iou[i] = val_iou[i] / len(val_dataloader.dataset) * batch_size * adjust_factor_val



        train_f1_macro = train_f1_macro / len(train_dataloader.dataset) * batch_size
        val_loss = val_loss / len(val_dataloader) * batch_size
        val_f1_macro = val_f1_macro / len(val_dataloader.dataset) * batch_size


        # store loss and dice score s
        train_loss_list.append(train_loss)
        for i in range(num_classes):
          train_dice_list[i].append(train_dice[i])
          train_iou_list[i].append(train_iou[i])
          val_dice_list[i].append(val_dice[i])
          val_iou_list[i].append(val_iou[i])
        train_f1_macro_list.append(train_f1_macro)
        val_loss_list.append(val_loss)
        val_f1_macro_list.append(val_f1_macro)

        # print epoch results
        print(f"--> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"--> Train F1_macro: {train_f1_macro:.4f} | Val F1_macro: {val_f1_macro:.4f}")
        for i in range(num_classes):
          print(f"--> Train Dice {i}: {train_dice[i]:.4f} | Val Dice {i}: {val_dice[i]:.4f}")
          print(f"--> Train IoU {i}: {train_iou[i]:.4f} | Val IoU {i}: {val_iou[i]:.4f}")

        # log to wandb
        if wandb_log:
            wandb_dict = {
                "epoch": (epoch+1),
                "train_loss": train_loss,
                "train_f1_macro": train_f1_macro,
                "val_loss": val_loss,
                "val_f1_macro": val_f1_macro,
                } | {
                    f"train_dice_{i}": train_dice[i] for i in range(num_classes)
                } | {
                    f"train_iou_{i}": train_iou[i] for i in range(num_classes)
                } | {
                    f"val_dice_{i}": val_dice[i] for i in range(num_classes)
                } | {
                    f"val_iou_{i}": val_iou[i] for i in range(num_classes)
                }
            if segmentation_ouput:
                wandb_dict = wandb_dict | slices_dic
            wandb.log(wandb_dict)

        # save best model
        if wandb_dict["val_dice_1"] > best_val_dice:
            patience_count = 0
            best_val_dice = wandb_dict["val_dice_1"]
            best_epoch = epoch
            torch.save(model.state_dict(), f"{run_dir}/best_model.pt")
            if wandb_log:
                wandb.save(f"{run_dir}/best_model.pt", base_path=run_dir)
        else:
            patience_count+=1

        if patience == patience_count:
          break

        #scheduler update
        scheduler.step(val_loss)

        # reset params
        train_loss = 0
        train_dice = [0 for i in range(num_classes)]
        train_iou = [0 for i in range(num_classes)]
        train_f1_macro = 0
        val_loss = 0
        val_dice = [0 for i in range(num_classes)]
        val_iou = [0 for i in range(num_classes)]
        val_f1_macro = 0
        slices_dic = {}

    if wandb_log:
        val_dice_max = {f"max_val_dice_{i}":max(val_dice_list[i]) for i in range(num_classes)}
        wandb.log(val_dice_max)
        if artifact_log:
            artifact = wandb.Artifact(
                f"{model.__class__.__name__}", 
                type="model", 
                description=f"Model after epoch {best_epoch}"
            )
            artifact.add_file(f"{run_dir}/best_model.pt")
            wandb.log_artifact(artifact)

    return val_dice_max