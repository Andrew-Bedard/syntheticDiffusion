import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import os
import streamlit as st
import random
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix

torch.manual_seed(0)


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


def train(net, model_trainloader, device, epochs=4, print_every=2000, learning_rate=0.001, momentum=0.9,
          progress_callback=None, progress_bar=None):
    """
    Trains a PyTorch neural network using a cross entropy loss function and stochastic gradient descent optimizer.

    Args:
    - net: A PyTorch neural network model to train.
    - trainloader: A PyTorch DataLoader representing the training set.
    - device: A string indicating the device to use for training (e.g. 'cuda' or 'cpu').
    - epochs (optional): An integer indicating the number of epochs to train for (default 4).
    - print_every (optional): An integer indicating how often to print the loss during training (default 2000).
    - learning_rate (optional): A float indicating the learning rate for the optimizer (default 0.001).
    - momentum (optional): A float indicating the momentum for the optimizer (default 0.9).
    - progress_callback (optional): A function to call with the progress of the training (default None).
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(model_trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_every:.3f}')
                running_loss = 0.0

            # Update progress
            if progress_callback is not None:
                progress_callback(progress_bar, (epoch * len(model_trainloader) + i) / (epochs * len(model_trainloader)))

    print('Finished Training')


def plot_metrics(metrics_history):
    """
    Plots the performance metrics (accuracy, precision, recall, and F1 score) against the percentage
    of supplementary data added to the original dataset.

    Args:
        metrics_history (list): A list of dictionaries containing the performance metrics at each
                                step of dataset augmentation. Each dictionary should have the keys:
                                'accuracy', 'precision', 'recall', and 'f1_score'.

    """
    percentages = [0, 25, 50, 75, 100]
    plt.figure(figsize=(12, 8))

    # Plot total accuracy
    plt.plot(percentages, [metrics['accuracy'] for metrics in metrics_history], label='Accuracy')

    # Plot average per-class metrics
    plt.plot(percentages, [sum(metrics['precision']) / len(metrics['precision']) for metrics in metrics_history], label='Precision')
    plt.plot(percentages, [sum(metrics['recall']) / len(metrics['recall']) for metrics in metrics_history], label='Recall')
    plt.plot(percentages, [sum(metrics['f1_score']) / len(metrics['f1_score']) for metrics in metrics_history], label='F1 Score')

    plt.xlabel('Percentage of Supplementary Data Added')
    plt.ylabel('Metric Value')
    plt.title('Model Performance Metrics vs. Supplementary Data Added')
    plt.legend()
    plt.grid()
    plt.show()


def calculate_metrics(model, dataloader, device, num_classes=10):
    """
    Computes accuracy, per-class accuracy, precision, recall, and F1 score for a given model and dataset.

    Args:
        model (torch.nn.Module): The trained PyTorch model for evaluation.
        dataloader (torch.utils.data.DataLoader): The DataLoader containing the dataset to evaluate.
        device (torch.device): The device (CPU or GPU) on which to perform the evaluation.
        num_classes (int): The number of classes in the classification problem.

    Returns:
        dict: A dictionary containing the following keys and their corresponding values:
              - 'accuracy': Total accuracy.
              - 'per_class_accuracy': A list with per-class accuracy.
              - 'precision': A list with per-class precision.
              - 'recall': A list with per-class recall.
              - 'f1_score': A list with per-class F1 score.
    """

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    precision = [cm[i, i] / sum(cm[:, i]) for i in range(num_classes)]
    recall = [cm[i, i] / sum(cm[i, :]) for i in range(num_classes)]
    f1_score = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(num_classes)]

    total_accuracy = sum([cm[i, i] for i in range(num_classes)]) / sum(sum(cm))

    metrics = {
        'accuracy': total_accuracy,
        'per_class_accuracy': [cm[i, i] / sum(cm[i, :]) for i in range(num_classes)],
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return metrics


def resize_normalize(img_path, resize=False):
    """ Resize 512 by 512 rgb images that we get from stable diffusion, convert to pytorch tensor, then normalize
    """

    if resize:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
        ])

    image = Image.open(img_path)  # Load your image
    image = transform(image)  # Apply the transform
    return (image)


def cfar_transform(cfar_img):
    """
    Transform and normalize an image from the CIFAR-10 dataset using PyTorch transforms.

    Args:
        cfar_img (PIL Image or numpy.ndarray): The image to be transformed.

    Returns:
        tensor: The transformed image as a PyTorch tensor, with pixel values
        normalized to the range [-1, 1]. The tensor has shape (3, H, W), where H and
        W are the height and width of the original image, respectively.

    Notes:
        This function applies the following transformations to the input image in
        sequence: convert the image to a PyTorch tensor, then normalize the pixel
        values of each color channel to have mean 0.5 and standard deviation 0.5.
        The resulting tensor has shape (3, H, W), where 3 corresponds to the three
        color channels (red, green, and blue).

        This function assumes that the input image has already been loaded using
        PyTorch's ImageFolder dataset, and is therefore in PIL Image format or
        numpy.ndarray format.

    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return (transform(cfar_img))


def load_and_transform_images(img_dir):
    """
    Loads and transforms images from a specified directory.

    Args:
        img_dir (str): The directory containing the images.

    Returns:
        torch.Tensor: A tensor containing the processed images.
    """
    # Get a list of the image paths
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]

    # Process each image and store the result in a list
    new_images = [resize_normalize(img_path) for img_path in img_paths]

    # Convert the list of images to a tensor
    new_images = torch.stack(new_images)

    return new_images


class Net(nn.Module):
    """
    A simple neural network architecture for image classification with three convolutional layers.

    The architecture consists of three convolutional layers followed by three fully connected layers. The input to
    the network should be a tensor of shape (batch_size, 3, height, width), where height and width are the dimensions of
    the input image, and the output should be a tensor of shape (batch_size, 10), where each element represents the
    predicted score for one of the 10 classes in the CIFAR-10 dataset.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        pool (nn.MaxPool2d): The max pooling layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        conv3 (nn.Conv2d): The third convolutional layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The output layer.

    Methods:
        forward(x): Defines the forward pass of the network.

    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 1 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (tensor): The input tensor, of shape (batch_size, 3, height, width).

        Returns:
            tensor: The output tensor, of shape (batch_size, 10).

        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomDataset(torch.utils.data.Dataset):
    """
    CustomDataset class that extends the torch.utils.data.Dataset class.

    Args:
        images (list): A list of image data.
        labels (list): A list of labels corresponding to each image.

    Attributes:
        images (list): A list of image data.
        labels (list): A list of labels corresponding to each image.

    Methods:
        __getitem__(self, index): Retrieves an image and label corresponding to the given index.
        __len__(self): Returns the total number of images in the dataset.
    """

    def __init__(self, images, labels):
        """
        Initializes a new CustomDataset instance with the given images and labels.
        """
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        """
        Retrieves the image and label corresponding to the given index.

        Args:
            index (int): The index of the image and label to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.images)


class CIFAR10Subsample(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class that subsamples the CIFAR-10 dataset.

    Args:
        root (str): The root directory where the dataset should be stored.
        train (bool, optional): If True, use the training dataset; if False, use the test dataset. Defaults to True.
        transform (callable, optional): A function or transform that takes in a PIL image and returns a transformed version. Defaults to None.
        percentage (int, optional): The percentage of the original dataset to include in the subsampled dataset. Defaults to 10.
        download (bool, optional): If True, download the dataset if it is not already present in the specified root directory. Defaults to True.
        seed (int, optional): The random seed to use when selecting samples. Defaults to None.

    Example:
        >>> cifar10_subsample = CIFAR10Subsample(root='./data', train=True, transform=transforms.ToTensor(), percentage=10)
        >>> dataloader = torch.utils.data.DataLoader(cifar10_subsample, batch_size=32, shuffle=True, num_workers=2)
    """

    def __init__(self, root, train=True, transform=None, percentage=10, download=True, seed=None):
        self.cifar_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download
        )

        random.seed(seed)
        num_samples = int(len(self.cifar_dataset) * percentage / 100)
        self.indices = random.sample(range(len(self.cifar_dataset)), num_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.cifar_dataset[self.indices[index]]

    def __init__(self, root, train=True, transform=None, percentage=10, download=True, seed=None,
                 class_proportions=None):
        self.cifar_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=download
        )

        self.class_labels = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }
        self.reverse_class_labels = {v: k for k, v in self.class_labels.items()}

        random.seed(seed)
        num_samples = int(len(self.cifar_dataset) * percentage / 100)

        if class_proportions is not None:
            self.indices = self._get_proportional_indices(class_proportions, num_samples)
        else:
            self.indices = random.sample(range(len(self.cifar_dataset)), num_samples)

    def _get_proportional_indices(self, class_proportions, num_samples):
        class_counts = {k: int(num_samples * v) for k, v in class_proportions.items()}
        indices_by_class = {k: [] for k in self.reverse_class_labels.values()}

        for i, (_, label) in enumerate(self.cifar_dataset):
            indices_by_class[label].append(i)

        proportional_indices = []
        for class_name, count in class_counts.items():
            class_idx = self.reverse_class_labels[class_name]
            proportional_indices += random.sample(indices_by_class[class_idx], count)

        random.shuffle(proportional_indices)
        return proportional_indices


def class_distribution_df(cifar10_subsample: CIFAR10Subsample) -> pd.DataFrame:
    """
    Create a DataFrame containing class counts for a CIFAR10Subsample dataset.

    Args:
        cifar10_subsample (CIFAR10Subsample): A CIFAR10Subsample dataset.

    Returns:
        pd.DataFrame: A DataFrame with columns "Class" and "Count".
    """

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Count the occurrences of each class in the dataset
    class_counts = Counter([cifar10_subsample[i][1] for i in range(len(cifar10_subsample))])

    # Replace the numeric class labels with class names
    class_counts_with_names = {class_names[k]: v for k, v in class_counts.items()}

    # Convert the class counts to a DataFrame
    class_counts_df = pd.DataFrame(list(class_counts_with_names.items()), columns=["Class", "Count"]).sort_values(
        "Class")

    return class_counts_df


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


# Define a callback function to update the progress bar
def update_progress(progress_bar, progress):
    progress_bar.progress(progress)


def train_base_cifar_model():
    """
    Trains the base CIFAR-10 model using the full CIFAR-10 dataset and stores the model, its accuracy,
    and per-class accuracy in the Streamlit session state. Displays the training progress and accuracy
    in the Streamlit app.
    """
    cifar_net = Net()
    cifar_net.to(device)

    # Display a message while the model is being trained
    training_message = st.empty()
    training_message.text("Training base CIFAR-10 model...")

    # Create a progress bar
    progress_bar = st.progress(0)

    st.session_state.cifar_net = Net()
    st.session_state.cifar_net.to(device)
    train(st.session_state.cifar_net, trainloader, device, epochs=epochs, print_every=4000, learning_rate=0.001,
          momentum=0.9, progress_callback=update_progress, progress_bar=progress_bar)

    # Remove the progress bar after training is complete
    progress_bar.empty()

    # Clear the message and display the results
    training_message.empty()
    cifar_metrics = calculate_metrics(st.session_state.cifar_net, testloader, device)
    st.session_state.cifar_net_accuracy = cifar_metrics['accuracy']

    st.write(f"CIFAR-10 model accuracy: {st.session_state.cifar_net_accuracy*100:.2f}%")


@st.cache_resource
def load_cifar10(percentage=10, cat_proportion=0.1):
    """

    :param percentage:
    :param cat_proportion:
    :return:
    """

    def normalize_class_proportions(class_proportions, target_class, new_proportion):
        """

        :param class_proportions:
        :param target_class:
        :param new_proportion:
        :return:
        """
        adjusted_proportions = class_proportions.copy()
        adjusted_proportions[target_class] = new_proportion

        sum_proportions = sum(adjusted_proportions.values())
        normalized_proportions = {key: value / sum_proportions for key, value in adjusted_proportions.items()}

        return normalized_proportions


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    class_proportions = {'airplane': 0.1,
                         'automobile': 0.1,
                         'bird': 0.1,
                         'cat': cat_proportion,  # Pass the cat_proportion here
                         'deer': 0.1,
                         'dog': 0.1,
                         'frog': 0.1,
                         'horse': 0.1,
                         'ship': 0.1,
                         'truck': 0.1
                         }

    # Normalize the class proportions based on the new cat_proportion
    normalized_proportions = normalize_class_proportions(class_proportions, 'cat', cat_proportion)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    trainset = CIFAR10Subsample(root='./data', train=True, transform=transform, percentage=percentage,
                                class_proportions=class_proportions)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    return trainset, trainloader, testset, testloader


st.markdown("# Synthetic Diffusion")
st.markdown("## What's it all about")
st.markdown("""Welcome to this demonstration of using advanced techniques in artificial intelligence to generate 
synthetic data for practical applications in data science. In this project, we'll be showcasing the power of Stable 
Diffusion to generate synthetic data for a Convolutional Neural Network (CNN) image classification problem. By 
augmenting our training data with synthetic images, we aim to improve the classification accuracy of the model. We've 
chosen the CIFAR-10 dataset for this demonstration as it's lightweight and well-suited for our purposes. The CIFAR-10 
dataset is widely recognized in the computer vision community, making it an ideal choice to demonstrate the 
effectiveness of using synthetic data generated by Stable Diffusion.""")

st.markdown("## CIFAR-10")
st.markdown(" Let's take a look at the CIFAR-10 dataset")
# Load the image using PIL
image_path = 'C:\\Users\\andre\\Projects\\syntheticDiffusion\\figures\\cifar10.png'
image = Image.open(image_path)

# Display the image using Streamlit
st.image(image, caption='CIFAR-10 dataset example', use_column_width=True)

st.markdown("## Methods")
st.markdown("""
The process we'll be following consists of three main steps:

1. **Synthetic Data Generation**: We'll begin by generating a synthetic dataset of cat images using Stable Diffusion, 
a powerful technique for creating realistic images that closely resemble their real-world counterparts. 

2. **Data Transformation**: After obtaining the synthetic cat dataset, we'll transform it to match the format of the 
CIFAR-10 dataset. This ensures that our CNN model can seamlessly process the synthetic data in conjunction with the 
original dataset. 

3. **Subsampling CIFAR-10**: To emphasize the impact of adding synthetic data to our model, we'll subsample the 
CIFAR-10 dataset, working with a smaller number of images than typically used. This simulates a situation where 
limited data is available, and the addition of synthetic images can provide a significant boost in classification 
performance.

By following these steps, we aim to demonstrate the potential benefits of incorporating synthetic data generated 
using Stable Diffusion into a CNN model for image classification. This demonstration will showcase the practical 
implications of using synthetic data to improve model performance in real-world applications.""")

st.markdown("""
Having generated a collection of realistic looking cat images we have to transform the format to match that of the
CIFAR-10 dataset. The most important change is that we have to downsample our 512x512 images to 32x32""")

# Load the image using PIL
image_path = 'C:\\Users\\andre\\Projects\\syntheticDiffusion\\figures\\cat_to_cat.png'
image = Image.open(image_path)

# Display the image using Streamlit
st.image(image, caption='Resize of synthetic cat images to CIFAR-10 format', use_column_width=True)


# Define some global variables

synthetic_imgDir = "C:\\Users\\andre\\Projects\\syntheticDiffusion\\data\\synthetic_cats\\"

# These are the classes in the cifar-10 dataset (in proper order)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # TODO: do I still need this?

batch_size = 16
num_classes = 10
epochs = 30
cifar_percentage = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # If gpu exists use cuda

# Load and transform synthetic images

custom_images = load_and_transform_images(synthetic_imgDir)
custom_labels = len(custom_images) * [3]  # Your assigned labels(cat)

trainset, trainloader, testset, testloader = load_cifar10(cifar_percentage)

# Let's take a look at the distribution of classes in our CIFAR-10 subsample
st.markdown("Let's take a look at the distribution of classes for our subsampled CIFAR-10 dataset")
class_counts_df = class_distribution_df(trainset)
fig = bar_chart_classes(class_counts_df)
st.pyplot(fig)
# st.bar_chart(class_counts_df.set_index("Class"))  # This is the previous version of the bar chart, which I like better but I don't know if it's possible to rotate the class labels

# Create a button to decrease cat samples

# # If the button is pressed, show the slider and store its value in the session state
# st.markdown("Let's take a look at the distribution of classes for our subsampled CIFAR-10 dataset")
# cat_proportion = st.slider("Proportion of cat images:", min_value=0.0, max_value=0.1, value=0.05, step=0.01)
# st.session_state.cat_proportion = cat_proportion
#
# # If the session state has a cat_proportion, use it to load the data and display the chart
# if "cat_proportion" in st.session_state:
#     trainset, trainloader, testset, testloader = load_cifar10(percentage=10, cat_proportion=st.session_state.cat_proportion)
#
#     st.markdown("Let's take a look at the distribution of classes for our subsampled dataset now")
#     class_counts_df = class_distribution_df(trainset)
#     fig = bar_chart_classes(class_counts_df)
#     st.pyplot(fig)

st.markdown("# DUMMY")
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.plotting import figure
from bokeh.layouts import column


def interactive_bar_chart(class_labels, initial_proportions):
    # Create a DataFrame with the class labels and initial proportions
    data = pd.DataFrame({"classes": class_labels, "proportions": initial_proportions})

    # Create a ColumnDataSource from the DataFrame
    source = ColumnDataSource(data)

    # Create the interactive bar chart
    plot = figure(x_range=class_labels, height=500, title="Class Proportions")
    plot.vbar(x="classes", top="proportions", width=0.9, source=source)
    plot.y_range.start = 0
    plot.xgrid.grid_line_color = None
    plot.xaxis.axis_label = "Classes"
    plot.yaxis.axis_label = "Proportions"

    # Create a slider for the cat class
    slider = Slider(start=0, end=0.1, value=0.1, step=0.01, title="Proportion of cat images")

    # Update the callback to store the current class proportions in the session state
    callback = CustomJS(args=dict(source=source, slider=slider), code="""
        const data = source.data;
        const cat_proportion = slider.value;

        // Calculate the sum of the proportions of other classes
        const other_classes_sum = data["proportions"].reduce((sum, value, index) => {
            return sum + (data["classes"][index] === "cat" ? 0 : value);
        }, 0);

        for (let i = 0; i < data["classes"].length; i++) {
            if (data["classes"][i] === "cat") {
                data["proportions"][i] = cat_proportion;
            } else {
                data["proportions"][i] = data["proportions"][i] * (1 - cat_proportion) / other_classes_sum;
            }
        }
        source.change.emit();

        // Store the current class proportions in the session state and trigger a re-run
        streamlit.setSessionState({class_proportions: data["proportions"], rerun: true});
    """)

    # Assign the callback to the slider
    slider.js_on_change("value", callback)

    # Display the interactive bar chart and the slider using Bokeh
    st.bokeh_chart(column(slider, plot))

class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
initial_proportions = [0.1] * 10

interactive_bar_chart(class_labels, initial_proportions)

# Check if the current class proportions are available in the session state
if "class_proportions" in st.session_state:
# Convert the current class proportions to a dictionary (replace with your actual class labels)
    current_class_proportions = dict(zip(class_labels, st.session_state.class_proportions))

    # Reload the dataset and display other visualizations using the current class proportions
    trainset, trainloader, testset, testloader = load_cifar10(percentage=10,
                                                              class_proportions=current_class_proportions)

    # Display any other visualizations or information related to the dataset
    # ...

    # Reset the 'rerun' flag in the session state after the app has re-run
    if "rerun" in st.session_state and st.session_state.rerun:
        st.session_state.rerun = False

if st.button('show cat class length for debugging'):
    # Let's take a look at the distribution of classes in our CIFAR-10 subsample
    st.markdown("Let's take a look at the distribution of classes for our subsampled CIFAR-10 dataset")
    class_counts_df = class_distribution_df(trainset)
    fig = bar_chart_classes(class_counts_df)
    st.pyplot(fig)

st.markdown("# End of DUMMY")

# Create a button for triggering the training process for base cifar-10 model
if st.button('Train CIFAR-10 base model'):
    train_base_cifar_model()

# ... (Load and transform synthetic images)
custom_images = load_and_transform_images(synthetic_imgDir)
custom_labels = len(custom_images) * [3]  # Your assigned labels(cat)

# Define slider for selecting number of custom images to add
st.markdown("""To further illustrate the performance increase that we get when we add more of our synthetic images
you can adjust the number of our synthetic cats you want to add to the CIFAR-10 dataset""")
#num_custom_images = st.slider('Select number of custom images to add to dataset', 0, len(custom_images), step=10)
num_custom_images = st.number_input(f"Select the number of custom images to add to dataset (max{len(custom_images)})",
                                    min_value=0,
                                    max_value=len(custom_images), step=10)

# Normalize CIFAR-10 images
cfar_images = [trainset[i][0] for i in range(len(trainset))]
cfar_images = torch.stack(cfar_images)

# Concatenate our synthetic images with CIFAR-10 images
custom_train_images = torch.cat((cfar_images, custom_images[:num_custom_images]), dim=0)

# Extract targets from the CIFAR-10 Subsample
cfar_targets = [trainset[i][1] for i in range(len(trainset))]
custom_train_labels = cfar_targets + custom_labels[:num_custom_images]

# Create a custom dataloader for our new combined dataset
custom_trainset = CustomDataset(custom_train_images, custom_train_labels)
custom_trainloader = torch.utils.data.DataLoader(custom_trainset, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)

# Initialize variables to store the previous custom model's accuracies in the session state
if 'prev_custom_net_accuracy' not in st.session_state:
    st.session_state.prev_custom_net_accuracy = None

if 'prev_custom_net_per_class_accuracy' not in st.session_state:
    st.session_state.prev_custom_net_per_class_accuracy = None

# Create a button for triggering the training process
if st.button('Train synthetically enhanced model'):

    # Createinstance of your model
    custom_net = Net()
    custom_net.to(device)

    # Display a message while the model is being trained
    training_message = st.empty()
    training_message.text("Training custom model...")

    # Create a progress bar
    progress_bar = st.progress(0)

    # Train the networks using the custom and CIFAR-10 dataloaders
    train(custom_net, custom_trainloader, device, epochs=epochs, print_every=4000, learning_rate=0.001, momentum=0.9,
          progress_callback=update_progress, progress_bar=progress_bar)

    # Remove the progress bar after training is complete
    progress_bar.empty()

    # Clear the message and display the results
    training_message.empty()

    metrics = calculate_metrics(custom_net, testloader, device, len(class_labels))

    # Display overall accuracy
    st.subheader("Overall Model Accuracy")
    st.write(f"Custom model accuracy: {metrics['accuracy']*100:.2f}%")
    # st.write(f"CIFAR-10 model accuracy: {st.session_state.cifar_net_accuracy:.2f}%")

    # Plot per-class accuracy
    st.subheader("Per-class Model Accuracy")
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, metrics['per_class_accuracy']*100, width, label='Custom Model')
    rects2 = ax.bar(x + width / 2, st.session_state.cifar_net_per_class_accuracy*100, width, label='CIFAR-10 Model')

    ax.set_ylabel('Accuracy')
    ax.set_title('Per-class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.1f}%'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        rotation=75)  # Rotate the text by 45 degrees


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    st.pyplot(fig)

    # Compare the new custom model with the previous custom model if it exists
    if st.session_state.prev_custom_net_accuracy is not None:
        st.subheader("Comparison with the Previous Custom Model")
        st.write(f"Previous custom model accuracy: {st.session_state.prev_custom_net_accuracy*100:.2f}%")
        st.write(f"New custom model accuracy: {metrics['accuracy']*100:.2f}%")

        # Plot per-class accuracy comparison
        fig2, ax2 = plt.subplots()
        rects3 = ax2.bar(x - width / 2, st.session_state.prev_custom_net_per_class_accuracy, width,
                         label='Previous Custom Model')
        rects4 = ax2.bar(x + width / 2, metrics['per_class_accuracy']*100, width, label='New Custom Model')

        ax2.set_ylabel('Accuracy')
        ax2.set_title('Per-class Accuracy Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.legend()

        autolabel(rects3)
        autolabel(rects4)

        fig2.tight_layout()

        st.pyplot(fig2)

    # Store the previous custom model's accuracies before updating the current ones
    st.session_state.prev_custom_net_accuracy = metrics['accuracy']
    st.session_state.prev_custom_net_per_class_accuracy = metrics['per_class_accuracy']

# TODO: Add some image interactivity
# TODO: more optimization of NN architecture?
# TODO: See how easy it would be to save the cifar NN to avoid retraining it all the time
# TODO: split this project up into a few files and use imports to clean it up
