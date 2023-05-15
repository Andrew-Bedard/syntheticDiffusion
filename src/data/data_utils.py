"""
Functions related to data loading, preprocessing and dataset creation
"""

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random
import pandas as pd
from collections import Counter
import streamlit as st


def resize_normalize(img_path):
    """ Resize 512 by 512 rgb images that we get from stable diffusion, convert to pytorch tensor, then normalize
    """

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])

    image = Image.open(img_path)  # Load your image
    image = transform(image)  # Apply the transform
    return image

def synthetic_load(img_path, resize=False):
    """

    :param img_path:
    :param resize_normalize:
    :return:
    """

    if resize:
        return resize_normalize(img_path)

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])

    image = Image.open(img_path)  # Load your image
    image = transform(image)  # Apply the transform
    return image

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

    return transform(cfar_img)


def load_and_transform_images(img_dir, resize=False):
    """
    Loads and transforms images from a specified directory.

    Args:
        img_dir (str): The directory containing the images.

    Returns:
        torch.Tensor: A tensor containing the processed images.
        :param resize:
    """
    # Get a list of the image paths
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')]

    # Process each image and store the result in a list
    new_images = [synthetic_load(img_path, resize) for img_path in img_paths]

    # Convert the list of images to a tensor
    new_images = torch.stack(new_images)

    return new_images


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


def subsample_dataset(dataset, percentage_per_class_dict):
    """
    Subsample the given dataset based on the percentage_per_class_dict.

    Args:
        dataset (torchvision.datasets.CIFAR10): The original dataset to subsample.
        percentage_per_class_dict (dict): A dictionary with class indices as keys and the percentage
                                          of samples to select per class as values.

    Returns:
        subsampled_images (list): List of subsampled image data.
        subsampled_labels (list): List of subsampled image labels.
    """

    # Calculate the maximum number of samples per class based on the percentage_per_class_dict
    max_counts = np.array(
        [int(len([label for _, label in dataset if label == i]) * percentage_per_class_dict[i]) for i in range(10)])

    # Initialize an array to keep track of the current count of samples per class
    counts = np.zeros(10, dtype=int)

    # Iterate through the dataset and add samples to the subsampled_data list if the count for that class has not reached its maximum
    subsampled_data = [(img, label) for img, label in dataset if counts[label] < max_counts[label]]
    counts = [counts[i] + 1 for i in range(10) if counts[i] < max_counts[i]]

    # Separate the images and labels from the subsampled_data list
    subsampled_images = [img for img, label in subsampled_data]
    subsampled_labels = [label for img, label in subsampled_data]

    return subsampled_images, subsampled_labels


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

class CustomCifar(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]  # Use self.targets instead of self.labels
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def targets(self):
        return [entry[1] for entry in self.data]  # Assuming label is at index 1

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
        class_proportions (dict, optional): A dictionary containing the desired class proportions. If not provided, equal proportions for all classes will be used.
        sampling_method (str, optional): The method to use for sampling the dataset. Can be either 'random' or 'uniform'. Defaults to 'random'.

    Example:
        >>> cifar10_subsample = CIFAR10Subsample(root='./data', train=True, transform=transforms.ToTensor(), percentage=10)
        >>> dataloader = torch.utils.data.DataLoader(cifar10_subsample, batch_size=32, shuffle=True, num_workers=2)
    """

    def __init__(self, root, train=True, transform=None, percentage=10, download=True, seed=None,
                 class_proportions=None, sampling_method='random'):
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
            if sampling_method == 'random':
                self.indices = random.sample(range(len(self.cifar_dataset)), num_samples)
            elif sampling_method == 'uniform':
                self.indices = self._get_uniform_indices(num_samples)
            else:
                # raise ValueError("Invalid sampling_method. Must be either 'random' or 'uniform'.")
                self.indices = self._get_uniform_indices(num_samples)  # I know this is redundant,
                # but I want to keep this last else loop in case I want to create new sampling methods

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

    def _get_uniform_indices(self, num_samples):
        num_classes = len(self.class_labels)
        samples_per_class = num_samples // num_classes
        remaining_samples = num_samples % num_classes

        indices_by_class = {k: [] for k in self.reverse_class_labels.values()}

        for i, (_, label) in enumerate(self.cifar_dataset):
            indices_by_class[label].append(i)

        uniform_indices = []
        for class_idx in self.reverse_class_labels.values():
            uniform_indices += random.sample(indices_by_class[class_idx], samples_per_class)

        # Add the remaining samples to the indices
        for i in range(remaining_samples):
            class_idx = i % num_classes
            uniform_indices += random.sample(indices_by_class[class_idx], 1)

        random.shuffle(uniform_indices)

        return uniform_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.cifar_dataset[self.indices[index]]

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

def get_transforms():
    """
    Returns a composed transform object to be applied on the CIFAR-10 dataset.

    Returns:
        transforms.Compose: A transform object that applies a series of transformations.
    """
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def create_trainloader(trainset, batch_size):
    """
    Creates a DataLoader for the training set.

    Args:
        trainset (Dataset): The training dataset.
        batch_size (int): The number of samples per batch to load.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the training set.
    """
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

def create_testloader(testset, batch_size):
    """
    Creates a DataLoader for the test set.

    Args:
        testset (Dataset): The test dataset.
        batch_size (int): The number of samples per batch to load.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the test set.
    """
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

@st.cache_resource
def load_cifar10_trainset(percentage=10, batch_size=16, class_proportions=None, sample_method=None):
    """
    Loads the CIFAR-10 training dataset with a custom proportion of cat samples.

    Args:
        percentage (int, optional): The percentage of the total dataset to use for the training set. Defaults to 10.
        cat_proportion (float, optional): The proportion of cat samples in the training set. Defaults to 0.1.
        batch_size (int, optional): The number of samples per batch to load. Defaults to 100.
        class_proportions (dict, optional): A dictionary containing the initial class proportions. If not provided,
                                            a default dictionary with equal proportions for all classes will be used.
        sample_method (str, optional): either 'random' or 'uniform', determines the method that's used to subsample data

    Returns:
        tuple: A tuple containing the trainset and trainloader for the modified CIFAR-10 dataset.
    """
    transform = get_transforms()
    # normalized_proportions = normalize_class_proportions(class_proportions, 'cat', cat_proportion)

    trainset = CIFAR10Subsample(root='./data', train=True, transform=transform, percentage=percentage,
                                sampling_method=sample_method)
    trainloader = create_trainloader(trainset, batch_size)

    return trainset, trainloader


def load_cifar10_testset(batch_size=16):
    """
    Loads the CIFAR-10 test dataset.

    Args:
        batch_size (int, optional): The number of samples per batch to load. Defaults to 100.

    Returns:
        tuple: A tuple containing the testset and testloader for the CIFAR-10 dataset.
    """
    transform = get_transforms()

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = create_testloader(testset, batch_size)

    return testset, testloader


def average_metrics(all_metrics, num_trials):
    """
    Computes the average evaluation metrics across all trials for each percentage of cat images in the training dataset.

    This function takes the evaluation metrics dictionary produced by the `train_and_evaluate_models` function and
    calculates the average metrics across all trials for each percentage of cat images.

    Args:
        all_metrics (dict): A dictionary containing the evaluation metrics for each model, keyed by the percentage
                            of cat images in the training dataset. Each value in the dictionary is a list of metrics
                            dictionaries, one for each trial.
        num_trials (int): The number of trials for each model.

    Returns:
        averaged_metrics (dict): A dictionary containing the averaged evaluation metrics for each model, keyed by the
                                 percentage of cat images in the training dataset. Each value in the dictionary is a
                                 metrics dictionary with averaged values for accuracy, per_class_accuracy, precision,
                                 recall, and f1_score.
    """
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



