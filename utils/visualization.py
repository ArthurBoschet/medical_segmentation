import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

def visualize_positional_encodings(pos_encoding, figsize=(15,10), title="Positional Encodings"):
    '''
    shows a visualization of a positional encoding in a tensor
    Parameters:
        pos_encoding (torch.tensor): positional encoding tensor (C, D, H, W)
        figsize (Tuple[int]): size of the figure
        title (str): title of the plot
    '''

    @interact
    def visualise_4d_image(d=(0,pos_encoding.shape[1]-1), h=(0,pos_encoding.shape[2]-1), w=(0,pos_encoding.shape[3]-1)): 
        #get image
        img = pos_encoding[:, d, h, w].reshape(1, -1)

        #plot slice
        plt.figure(figsize=figsize)
        plt.imshow(img, aspect=pos_encoding.shape[0]/6)
        plt.xlabel('channels')
        plt.clim(-1,1)

        #show plot
        plt.show()

def visualize_attention(attention_weights_avg, figsize=(15,15), fraction=0.02, pad=0.05, title='Average Attention Weights'):
    '''
    shows a visualization of a positional encoding in a tensor
    Parameters:
        pos_encoding (torch.tensor): positional encoding tensor (C, D, H, W)
        figsize (Tuple[int]): size of the figure
        fraction (float): colorbar parameter
        pad (float): colorbar padding from the figure
        title (str): title of the plot
    '''

    @interact
    def visualise_4d_image(
        image=(0,attention_weights_avg.shape[0]-1),
        depth_decoder=(0,attention_weights_avg.shape[-3]-1),
        height_decoder=(0,attention_weights_avg.shape[-2]-1),
        width_decoder=(0,attention_weights_avg.shape[-1]-1),
        depth_skip=(0,attention_weights_avg.shape[1]-1),
        ): 
        #get image
        img = attention_weights_avg[image, depth_skip, :, :, depth_decoder, height_decoder, width_decoder]

        #plot slice
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(img)
        fig.colorbar(im, fraction=fraction, pad=pad)
        plt.title(title)

        #show plot
        plt.show()


def visualize_dataloaders_overlap(dataloader, cmap='gray' , alpha = 0.3, figsize=(8, 8)):
    '''
    shows a visualization of a slice of an image and its label from a dataloader

    Args:
        dataloader: torch.utils.data.DataLoader
            dataloader to visualize
        cmap: str
            str for the style of cmap of the label image
        alpha: float
            % of  oppacity for the label
        figsize: tuple
            size of the figure
    '''

    @interact
    def plot_slice(image=(1, len(dataloader.dataset)),
                   slice=(1, dataloader.dataset[0][0][0].numpy().shape[0]),
                   ):
        # get image and label
        im = dataloader.dataset[image][0][0].numpy()
        label = dataloader.dataset[image][1][0].numpy()
        if slice > im.shape[0] - 1:
            slice = im.shape[0] - 1

        # plot slice
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(im[slice], cmap="gray")
        ax.imshow(label[slice], cmap="jet", alpha=alpha)
        ax.set_title(f"Image and label")

        # show plot
        plt.show()

def plot_learning_curves(df, metric, y_axis="epoch", figsize=(10, 5), show=False, save_path=None):
    """
    Plot the learning curves of a model for k-fold cross-validation training.

    Args:
        df (pandas.DataFrame):
            The logged data as a pandas DataFrame.
            Accepts list of dataframes.
        metric (str):
            The metric to plot.
        y_axis (str):
            The y-axis name on the plot.
        figsize (tuple):
            The figure size.
        show (bool):
            Whether to show the plot.
        save_path (str):
            The path to save the plot.
            If None, the plot is not saved.

    Returns:
        matplotlib.pyplot:
            The plot object.
    """
    if not isinstance(df, list):
        df = [df]
    train_mean = np.mean([fold_df[f"train_{metric}"] for fold_df in df], axis=0)
    train_std = np.std([fold_df[f"train_{metric}"] for fold_df in df], axis=0)
    val_mean = np.mean([fold_df[f"val_{metric}"] for fold_df in df], axis=0)
    val_std = np.std([fold_df[f"val_{metric}"] for fold_df in df], axis=0)
    plt.figure(figsize=figsize)
    plt.plot(train_mean, label="Training Score", color="blue", marker="o")
    plt.fill_between(np.arange(len(train_mean)), train_mean - train_std, train_mean + train_std, alpha=0.15, color="blue")
    plt.plot(val_mean, label="Cross Validation Score", color="red", marker="s")
    plt.fill_between(np.arange(len(val_mean)), val_mean - val_std, val_mean + val_std, alpha=0.15, color="red")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_axis)
    plt.title(f"Learning curve for {y_axis} on {len(df)} folds")
    plt.grid()
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    
    return plt