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

    Returns:
        fig (matplotlib.figure.Figure): A matplotlib figure containing the plotted metrics.

    """
    percentages = [0, 25, 50, 75, 100]
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot total accuracy
    ax.plot(percentages, [metrics['accuracy'] for metrics in metrics_history], label='Accuracy')

    # Plot average per-class metrics
    ax.plot(percentages, [sum(metrics['precision']) / len(metrics['precision']) for metrics in metrics_history], label='Precision')
    ax.plot(percentages, [sum(metrics['recall']) / len(metrics['recall']) for metrics in metrics_history], label='Recall')
    ax.plot(percentages, [sum(metrics['f1_score']) / len(metrics['f1_score']) for metrics in metrics_history], label='F1 Score')

    ax.set_xlabel('Percentage of Supplementary Data Added')
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Performance Metrics vs. Supplementary Data Added')
    ax.legend()
    ax.grid()

    return fig



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
    precision = [cm[i, i] / sum(cm[:, i]) if sum(cm[:, i]) != 0 else 0 for i in range(num_classes)]
    recall = [cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) != 0 else 0 for i in range(num_classes)]
    f1_score = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else
                0 for i in range(num_classes)]
    total_accuracy = sum([cm[i, i] for i in range(num_classes)]) / sum(sum(cm))

    metrics = {
        'accuracy': total_accuracy,
        'per_class_accuracy': [cm[i, i] / sum(cm[i, :]) if sum(cm[i, :]) != 0 else 0 for i in range(num_classes)],
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
    st.session_state.cifar_net_per_class_accuracy = cifar_metrics['per_class_accuracy']

    #st.write(f"CIFAR-10 model accuracy: {st.session_state.cifar_net_accuracy*100:.2f}%")


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


# These are the classes in the cifar-10 dataset (in proper order)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # TODO: do I still need this?

batch_size = 16
num_classes = 10
epochs = 30
cifar_percentage = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # If gpu exists use cuda

def page1():
    st.title("Background")
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
    image_path = '..\\syntheticDiffusion\\figures\\cifar10.png'
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
    image_path = '..\\syntheticDiffusion\\figures\\cat_to_cat.png'
    image = Image.open(image_path)

    # Display the image using Streamlit
    st.image(image, caption='Resize of synthetic cat images to CIFAR-10 format', use_column_width=True)

def page2():
    st.title("Generating Synthetic Images of cats")
    st.markdown("")
    st.markdown("[Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)", unsafe_allow_html=True)


def page3():

    st.markdown("WARNING! Pressing the following button will take a long time to complete")
    if st.button("Perform batch training/metrics"):

        trainset, trainloader, testset, testloader = load_cifar10(cifar_percentage)

        st.title("Pre-calculated results")

        from torch.utils.data import Subset
        import random

        def train_cifar_model_with_percentage(percentage, trainset, trainloader, device):
            # Find the indices of the cat images in the trainset
            cat_indices = [i for i, (image, label) in enumerate(trainset) if
                           label == 3]  # Assuming cat class has a label of 3

            # Calculate the number of cat images to include in the subsampled dataset
            num_cat_images = int(len(cat_indices) * (percentage / 100))

            # Randomly choose the cat images to include
            chosen_cat_indices = random.sample(cat_indices, num_cat_images)

            # Find the indices of the non-cat images in the trainset
            non_cat_indices = [i for i in range(len(trainset)) if i not in cat_indices]

            # Combine the chosen cat image indices with the non-cat image indices
            new_trainset_indices = non_cat_indices + chosen_cat_indices

            # Create a new Subset with the specified percentage of cat images
            new_trainset = Subset(trainset, new_trainset_indices)

            # Create a new DataLoader with the modified trainset
            new_trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=trainloader.batch_size, shuffle=True,
                                                          num_workers=trainloader.num_workers)

            # Train the model using the modified trainloader
            model = Net()
            model.to(device)
            train(model, new_trainloader, device, epochs=epochs, print_every=4000, learning_rate=0.001,
                  momentum=0.9)

            return model

        def train_and_evaluate_models(percentages, num_trials, trainset, trainloader, testloader, device):
            all_metrics = {p: [] for p in percentages}

            for p in percentages:
                print(p)
                for _ in range(num_trials):
                    model = train_cifar_model_with_percentage(p, trainset, trainloader, device)
                    metrics = calculate_metrics(model, testloader, device)
                    all_metrics[p].append(metrics)

            return all_metrics

        def average_metrics(all_metrics, num_trials):
            averaged_metrics = {}

            for p in all_metrics:
                averaged = {
                    'accuracy': 0,
                    'per_class_accuracy': np.zeros(10),
                    'precision': np.zeros(10),
                    'recall': np.zeros(10),
                    'f1_score': np.zeros(10)
                }

                for metrics in all_metrics[p]:
                    averaged['accuracy'] += metrics['accuracy']
                    averaged['per_class_accuracy'] += np.array(metrics['per_class_accuracy'])
                    averaged['precision'] += np.array(metrics['precision'])
                    averaged['recall'] += np.array(metrics['recall'])
                    averaged['f1_score'] += np.array(metrics['f1_score'])

                averaged['accuracy'] /= num_trials
                averaged['per_class_accuracy'] /= num_trials
                averaged['precision'] /= num_trials
                averaged['recall'] /= num_trials
                averaged['f1_score'] /= num_trials

                averaged_metrics[p] = averaged

            return averaged_metrics

        percentages = [0, 25, 50, 75, 100]
        num_trials = 3

        # Assuming you have your trainset, trainloader, testloader, and device set up
        all_metrics = train_and_evaluate_models(percentages, num_trials, trainset, trainloader, testloader, device)
        averaged_metrics = average_metrics(all_metrics, num_trials)

        # Plot the metrics
        metrics_history = [averaged_metrics[p] for p in percentages]
        fig = plot_metrics(metrics_history)
        st.pyplot(fig)


def page4():
    trainset, trainloader, testset, testloader = load_cifar10(cifar_percentage)
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    st.title("Experiment with Synthetic Diffusion")

    # Let's take a look at the distribution of classes in our CIFAR-10 subsample
    st.markdown("CIFAR-10 is a highly curated dataset with 50,000 images in the training set that are all perfectly "
                "balanced. Because we are doing some training and experimentation locally, we might not want to wait "
                "for a model to train using 50,000 images, additionally we really want to showcase the difference in "
                "performance when we are in a situation where we don't have all the data we would like, so right away "
                "let's subsample the CIFAR-10 dataset uniformly to 10% of it's original size")
    st.markdown("Let's take a look at the distribution of classes for our subsampled CIFAR-10 dataset")
    class_counts_df = class_distribution_df(trainset)
    fig = bar_chart_classes(class_counts_df)
    st.pyplot(fig)

    st.markdown("""We should now benchmark this base CIFAR-10 model with this subsampled training set. You can either 
    choose to load the CIFAR-10 base model that was included in the repository, or train a new one yourself on the 
    training set that was created to produce the previous bar char""")

    # Check if the cifar10_base_model.pt exists
    model_exists = os.path.exists("..\\syntheticDiffusion\\data\\models\\cifar10_base_model.pt")

    # Create two columns for placing the buttons side by side
    col1, col2 = st.columns(2)

    # Display the button for training the base model in the first column
    if col1.button('Train CIFAR-10 base model'):
        train_base_cifar_model()

    # If the base model file exists, display the button for using the existing model in the second column
    if model_exists:
        if col2.button('Use existing CIFAR-10 base model'):
            # Instantiate the model and load the state_dict from the file
            st.session_state.cifar_net = Net()
            st.session_state.cifar_net.load_state_dict(torch.load("..\\syntheticDiffusion\\data\\models\\cifar10_base_model.pt"))
            st.session_state.cifar_net.to(device)

            # Calculate the metrics for the existing model
            cifar_metrics = calculate_metrics(st.session_state.cifar_net, testloader, device)
            st.session_state.cifar_net_accuracy = cifar_metrics['accuracy']
            st.session_state.cifar_net_per_class_accuracy = cifar_metrics['per_class_accuracy']

    # Show the net accuracy for the base model
    if 'cifar_net_accuracy' in st.session_state:
        st.write(f"CIFAR-10 model accuracy: {st.session_state.cifar_net_accuracy * 100:.2f}%")
        basecifar_fig = plot_single_per_class_accuracy(st.session_state.cifar_net_per_class_accuracy, labels, 'CFIAR-10 base model')
        st.pyplot(basecifar_fig)

    st.markdown("Great! We now have a base model trained on the subsampled CIFAR-10 dataset, and it should work "
                "reasonably well considering the simplicitly of our CNN and the size of the dataset. However, "
                "in real world applications we would be quite lucky to have a real dataset that is totally"
                "balanced across all class labels, so let's do some further ")

    # Create a button to decrease cat samples
    # If the button is pressed, show the slider and store its value in the session state
    st.markdown("Let's take a look at the distribution of classes for our subsampled CIFAR-10 dataset")
    cat_proportion = st.slider("Total proportion of cat images:", min_value=0.0, max_value=0.09, value=0.05, step=0.01)
    st.session_state.cat_proportion = cat_proportion
    st.session_state.cat_num = cat_proportion * 10 * 500  # TODO: this should not be hardcoded like it is

    #
    # If the session state has a cat_proportion, use it to load the data and display the chart
    if "cat_proportion" in st.session_state:
        trainset, trainloader, testset, testloader = load_cifar10(percentage=10,
                                                                  cat_proportion=st.session_state.cat_proportion)

        st.markdown("Let's take a look at the distribution of classes for our subsampled dataset now")
        class_counts_df = class_distribution_df(trainset)
        # TODO: maybe this chart can be replaced by a bokeh one
        fig = bar_chart_classes(class_counts_df)
        st.pyplot(fig)

        # We don't want the load_cifar10 function to run again unless there has been a change in cat_proportion,
        # so we delete the session state for the variable after using it
        del st.session_state.cat_proportion

    st.markdown("""Depending on your choice of the proportion of cat images, we should have an unbalanced dataset 
    with fewer cat images than any other class. This is a situation we would normally try to avoid, so obviously, 
    we should start adding our own synthetically created cat images to the training set to suppliment it!""")

    # ... (Load and transform synthetic images)
    synthetic_imgDir = "..\\syntheticDiffusion\\data\\synthetic_cats\\"
    custom_images = load_and_transform_images(synthetic_imgDir)
    custom_labels = len(custom_images) * [3]  # Your assigned labels(cat)

    # Define slider for selecting number of custom images to add
    st.markdown("""Use the slider below to set the number of synthetic cat images to add to the training data. 
    By default it's set to fill the missing number of cat images, however you can add as many or few as you'd like. 
    We can compare the relative performance by trying multiple values""")
    num_custom_images = st.slider('Select number of custom images to add to dataset', 0,
                                  (len(custom_images) - len(custom_images)%50), value=(500 - int(st.session_state.cat_num)), step=50)

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
        train(custom_net, custom_trainloader, device, epochs=epochs, print_every=4000, learning_rate=0.001,
              momentum=0.9,
              progress_callback=update_progress, progress_bar=progress_bar)

        # Remove the progress bar after training is complete
        progress_bar.empty()

        # Clear the message and display the results
        training_message.empty()

        metrics = calculate_metrics(custom_net, testloader, device, len(classes))

        # Display overall accuracy
        st.subheader("Overall Model Accuracy")
        st.write(f"Custom model accuracy: {metrics['accuracy'] * 100:.2f}%")
        # st.write(f"CIFAR-10 model accuracy: {st.session_state.cifar_net_accuracy:.2f}%")

        # Plot per-class accuracy
        st.subheader("Per-class Model Accuracy")


        fig = plot_per_class_accuracy(metrics, st.session_state.cifar_net_per_class_accuracy, labels,
                                      ['Custom model', 'CIFAR-10 Base model'])

        st.pyplot(fig)

        # Compare the new custom model with the previous custom model if it exists
        if st.session_state.prev_custom_net_accuracy is not None:
            st.subheader("Comparison with the Previous Custom Model")
            fig2 = plot_per_class_accuracy(metrics, st.session_state.prev_custom_net_per_class_accuracy, labels,
                                           ['Current Custom Model', 'Previous Custom Model'])

            fig2.tight_layout()

            st.pyplot(fig2)

        # Store the previous custom model's accuracies before updating the current ones
        st.session_state.prev_custom_net_accuracy = metrics['accuracy']
        st.session_state.prev_custom_net_per_class_accuracy = metrics['per_class_accuracy']

def page5():

    # This page exists for debugging purposes

    import torch
    import io

    # ... your existing code ...

    if 'cifar_net_accuracy' in st.session_state:
        st.write(f"CIFAR-10 model accuracy: {st.session_state.cifar_net_accuracy * 100:.2f}%")

        # Save the model to a binary format (PyTorch)
        model_binary = io.BytesIO()
        torch.save(st.session_state.cifar_net.state_dict(), model_binary)
        model_binary.seek(0)

        # Create a download button for the model
        st.download_button(
            label="Download CIFAR-10 base model",
            data=model_binary,
            file_name="cifar10_base_model.pt",
            mime="application/octet-stream"
        )


# Sidebar menu for navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Introduction", "Synthetic Cats", "Pre-calculated results", "Hands-on", "Debugging"])

st.markdown("# Synthetic Diffusion")

# Mapping of pages to their respective functions
pages = {
    "Introduction": page1,
    "Synthetic Cats": page2,
    "Pre-calculated results": page3,
    "Hands-on": page4,
    "Debugging": page5
}

# Call the function corresponding to the selected page
pages[selected_page]()


# TODO: add slider to reduce number of cat pictures
# TODO: add precalculated results to page2
# TODO: add the ability to check for cifar10 base model before retraining, but also allow for training
