import os
import wandb
import shutil
import subprocess
import pandas as pd

from tqdm import tqdm


def get_wandb_run_data(entity, project, run_id):
    """
    Retrieve the logged data from a WandB run.
    
    Args:
        entity (str): 
            The entity name.
        project (str):
            The project name.
        run_id (str):
            The run id.

    Returns:
        history_df (pandas.DataFrame):
            The logged data as a pandas DataFrame.
    """
    # define the project and run you want to retrieve data for
    run_path = os.path.join(entity, project, run_id)

    # initialize the wandb run object
    api = wandb.Api()
    run = api.run(run_path)

    # get the history of the run
    history = run.history()

    # convert the history to a pandas dataframe
    history_df = pd.DataFrame(history)

    # remove useless columns
    history_df = history_df[[col for col in history_df.columns if col.startswith("train") or col.startswith("val")]]

    # remove all rows with nan values
    history_df = history_df.dropna()

    # reset index to start at 0
    history_df = history_df.reset_index(drop=True)

    return history_df


def sync_offline_runs(folder_path, delete=False):
    """
    Sync offline wandb runs to the cloud.
    
    Args:
        folder_path (str):
            The path to the folder containing the offline wandb runs.
        delete (bool):
            Whether to delete the offline runs after syncing.
    """
    # get all wandb folders
    wandb_folders = [os.path.join(folder_path, folder) for folder in sorted(os.listdir(folder_path)) if folder.startswith("offline-run-")]

    # sync each wandb folder
    for wandb_folder in tqdm(wandb_folders):
        # check if run is finished
        if os.path.exists(os.path.join(wandb_folder, "files", "wandb-summary.json")):
            # sync wandb folder
            subprocess.run(["wandb", "sync", wandb_folder])

            # delete wandb folder
            if delete:
                shutil.rmtree(wandb_folder)
