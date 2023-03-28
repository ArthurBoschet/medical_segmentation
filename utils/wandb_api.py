import os
import wandb
import pandas as pd


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