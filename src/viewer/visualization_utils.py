"""
Functions for visualizing images and plots
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import os
import random

from PIL import Image


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
    import plotly.graph_objects as go
    """
    Create a bar chart for all the classes of the CIFAR10Subsample object using Plotly.

    :param df: DataFrame containing 'Class' and 'Count' columns.
    :return: Plotly Figure object.
    """
    fig = go.Figure(data=go.Bar(x=df["Class"], y=df["Count"]))
    fig.update_layout(
        title="Class Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        xaxis_tickangle=-45,  # Rotate class labels 45 degrees
        height=600  # Set the height of the figure
    )

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


def create_interactive_plot(df):
    """
    Creates an interactive Plotly figure showing the comparison of various metrics
    (accuracy, per_class_accuracy, precision, recall, f1_score) for 'real' and 'synthetic' data sources.

    Each metric is represented by a trace for each data source. By default, only the 'accuracy' metric is visible.
    The figure includes a dropdown menu to select the metric to display.

    This function is specifically designed for usage in a Streamlit application, where it can provide an interactive
    visualization for exploring the performance of two models under different proportions of real or synthetic data.

    Parameters:
    df (pandas.DataFrame): A DataFrame containing the performance metrics.
                          The DataFrame should contain the following columns:
                          'source' (with values 'real' or 'synthetic'),
                          'metric' (with values 'accuracy', 'per_class_accuracy', 'precision', 'recall', 'f1_score'),
                          'percentage' (representing the proportion of data used),
                          'value' (representing the metric value).

    Returns:
    plotly.graph_objs._figure.Figure: The interactive Plotly figure.
    """

    import plotly.graph_objs as go

    # Create a Plotly figure
    fig = go.Figure()

    # Define the metrics
    metrics = ['accuracy', 'per_class_accuracy', 'precision', 'recall', 'f1_score']

    # Add traces for each source and metric
    for metric in metrics:
        for source in df['source'].unique():
            visible = metric == 'accuracy'
            fig.add_trace(go.Scatter(x=df[(df['source'] == source) & (df['metric'] == metric)]['percentage'],
                                     y=df[(df['source'] == source) & (df['metric'] == metric)]['value'],
                                     mode='lines+markers',
                                     name=f"{source} {metric}",
                                     visible=visible))

    # Update layout with dropdown menu for metrics
    fig.update_layout(
        title="Comparison of Metrics for Real and Synthetic",
        xaxis_title="Percentage",
        yaxis_title="Value",
        updatemenus=[
            go.layout.Updatemenu(
                buttons=list([
                    dict(label=metric,
                         method="update",
                         args=[{"visible": [m == metric for m in metrics for _ in range(2)]},
                               {"title": metric}])
                    for metric in metrics
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )

    return fig


def create_comparison_plot(df1, df2, metric_names, class_index):
    """
    Creates an interactive comparison plot of the specified performance metrics for 'real' and 'synthetic' data sources.

    This function first prepares the data by adding a 'source' column to each DataFrame, melting the DataFrames to a long format,
    and merging the two DataFrames. It then filters the rows based on the provided metric names, converts any mixed format strings
    to either a single float or a list of floats, and selects the values corresponding to the specified class index for all metrics
    except 'accuracy'. Finally, it calls `create_interactive_plot` on the prepared DataFrame.

    Parameters:
    df1 (pandas.DataFrame): A DataFrame containing the performance metrics for the 'real' data source.
    df2 (pandas.DataFrame): A DataFrame containing the performance metrics for the 'synthetic' data source.
    metric_names (list of str): The names of the metrics to include in the plot.
    class_index (int): The index of the class to select for metrics that are presented as lists.

    Returns:
    plotly.graph_objs._figure.Figure: The interactive Plotly figure.
    """
    # Add a 'source' column to identify the origin
    df1['source'] = 'Real'
    df2['source'] = 'Synthetic'

    # Melt the dataframes to a long format
    df1 = df1.melt(id_vars=['metric', 'source'], var_name='percentage', value_name='value')
    df2 = df2.melt(id_vars=['metric', 'source'], var_name='percentage', value_name='value')

    # Merge the two dataframes
    combined_df = pd.concat([df1, df2])

    # Filter rows based on the metric names
    #combined_df = combined_df[combined_df['metric'].isin(metric_names)]

    # Something weird is happening where the values in the dataframe are being saved as a string instead of a list,
    # let's convert these back into lists

    # Custom function to convert mixed format strings to either a single float or a list of floats
    def mixed_str_to_float(mixed_str):
        mixed_str = mixed_str.strip()
        if mixed_str.startswith('[') and mixed_str.endswith(']'):
            mixed_str = mixed_str[1:-1]  # Remove the brackets
            return [float(x) for x in mixed_str.split()]
        else:
            return float(mixed_str)

    # Apply the custom function to the DataFrame column
    combined_df['value'] = combined_df['value'].apply(mixed_str_to_float)

    # Modify this line in the `create_comparison_plot` function:
    combined_df.loc[combined_df['metric'] != 'accuracy', 'value'] = combined_df.loc[
        combined_df['metric'] != 'accuracy', 'value'].apply(lambda x: x[class_index])

    # Convert 'percentage' column to numeric type
    combined_df['percentage'] = pd.to_numeric(combined_df['percentage'])

    # Call the function with your DataFrame
    return create_interactive_plot(combined_df)


def show_images_in_grid(folder_path, num_images=9, grid_size=(3, 3)):
    """
    Displays a grid of randomly selected images in a Streamlit app.

    Parameters:
    folder_path (str): Path to the folder containing the images.
    num_images (int): Number of images to display.
    grid_size (tuple): Tuple specifying the layout of the grid.

    Returns:
    None
    """
    # Get a list of all image file paths
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   os.path.isfile(os.path.join(folder_path, f))]

    # Randomly select num_images files
    selected_images = random.sample(image_files, num_images)

    # Open the images and store them in a list
    images = [Image.open(img_path) for img_path in selected_images]

    # Create a subplot for the grid
    fig, axs = plt.subplots(*grid_size)

    # Loop through the grid and add the images
    for ax, img in zip(axs.flatten(), images):
        ax.imshow(img)
        ax.axis('off')  # Hide axes

    # Use st.pyplot to display the figure in Streamlit
    st.pyplot(fig)

