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


def plot_learning_curves(dfs, metric, model_names, y_axis, figsize=(10, 5), show=False, save_path=None):
    """
    Plot the learning curves of a model for k-fold cross-validation training.

    Args:
        dfs list(list(pandas.DataFrame)):
            The logged data as a list of list of pandas DataFrame.
            i.e. [model1[fold1_df, fold2_df, ...], model2[fold1_df, fold2_df, ...], ...]
        metric (str):
            The metric to plot.
        model_names list(str):
            List of names of the models used for training.
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
    assert isinstance(dfs, list)
    assert isinstance(dfs[0], list)
    assert isinstance(model_names, list)

    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    plt.figure(figsize=figsize)
    for i, df in enumerate(dfs):
        mean = np.mean([fold_df[metric] for fold_df in df], axis=0)
        std = np.std([fold_df[metric] for fold_df in df], axis=0)
        plt.plot(mean, label=model_names[i], color=colors[i])
        plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.15, color=colors[i])
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(y_axis)
    plt.title(f"{y_axis} on {len(df)} folds")
    plt.rc('font', size=14)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    
    return plt
