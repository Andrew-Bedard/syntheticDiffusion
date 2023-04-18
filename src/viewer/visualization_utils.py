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


def plot_metrics(metrics_history):
    """
    Plots the performance metrics (accuracy, precision, recall, and F1 score) against the percentage
    of supplementary data added to the original dataset.

    Args:
        metrics_history (list): A list of dictionaries containing the performance metrics at each
                                step of dataset augmentation. Each dictionary should have the keys:
                                'accuracy', 'precision', 'recall', and 'f1_score'.

    Returns:
        fig (matplotlib.figure.Figure): A matplotlib figure containing the plotted metrics.

    """
    percentages = [0, 25, 50, 75, 100]
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot total accuracy
    ax.plot(percentages, [metrics['accuracy'] for metrics in metrics_history], label='Accuracy')

    # Plot average per-class metrics
    ax.plot(percentages, [sum(metrics['precision']) / len(metrics['precision']) for metrics in metrics_history],
            label='Precision')
    ax.plot(percentages, [sum(metrics['recall']) / len(metrics['recall']) for metrics in metrics_history],
            label='Recall')
    ax.plot(percentages, [sum(metrics['f1_score']) / len(metrics['f1_score']) for metrics in metrics_history],
            label='F1 Score')

    ax.set_xlabel('Percentage of Supplementary Data Added')
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Performance Metrics vs. Supplementary Data Added')
    ax.legend()
    ax.grid()

    return fig

def bar_chart_classes(df):
    """
    Create a bar chart for all the classes of the CIFAR10Subsample object
    :param df:
    :return:
    """
    fig, ax = plt.subplots()
    ax.bar(df["Class"], df["Count"])
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")

    # Rotate the class labels 45 degrees
    plt.xticks(rotation=45)

    return fig

def autolabel(rects, ax, rotation=0):
    """Add labels to bars in a bar chart with a specified rotation.

    Args:
    rects: List of bar chart rectangles.
    ax: The axis object to annotate.
    rotation: Text rotation in degrees (default is 0).
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=rotation)


def plot_per_class_accuracy(metrics, cifar_net_per_class_accuracy, labels, model_labels):
    """Plot per-class accuracy for two models and display their percentage values on the bars.

    Args:
    metrics: Dictionary of custom model metrics.
    cifar_net_per_class_accuracy: Per-class accuracy values for the CIFAR-10 model.
    labels: List of class labels.
    model_labels: List of str, colour coding for bars to indicate model
    """
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, np.array(metrics['per_class_accuracy']) * 100, width, label=model_labels[0])
    rects2 = ax.bar(x + width / 2, np.array(cifar_net_per_class_accuracy) * 100, width, label=model_labels[1])

    autolabel(rects1, ax, rotation=65)
    autolabel(rects2, ax, rotation=65)

    ax.set_ylabel('Accuracy')
    ax.set_title('Per-class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(0, 100)
    ax.legend()

    return fig


def plot_single_per_class_accuracy(per_class_accuracy, labels, model_label):
    """Plot per-class accuracy for one model and display the percentage values on the bars.

    Args:
    metrics:
    labels: List of class labels.
    model_label: str, colour coding for bars to indicate model
    """
    x = np.arange(len(labels))
    width = 0.85

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, np.array(per_class_accuracy) * 100, width, label=model_label)

    autolabel(rects1, ax, rotation=65)

    ax.set_ylabel('Accuracy')
    ax.set_title('Per-class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(0, 100)
    ax.legend()

    return fig