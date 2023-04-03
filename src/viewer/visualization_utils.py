"""
Functions for visualizing images and plots
"""
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    """
    Display an image in a matplotlib plot.

    Args:
        img (tensor): The image to be displayed, as a PyTorch tensor. The tensor
        should have shape (3, H, W), where H and W are the height and width of the
        image, respectively.

    Returns:
        None.

    Notes:
        This function assumes that the input image has been normalized using the
        transforms.Normalize method with mean (0.5, 0.5, 0.5) and standard deviation
        (0.5, 0.5, 0.5). The function first unnormalizes the pixel values by
        scaling them by 2 and adding 0.5. It then converts the tensor to a numpy
        array and transposes the dimensions so that the color channels are the
        last dimension, in RGB order. Finally, it displays the image in a
        matplotlib plot.

    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()