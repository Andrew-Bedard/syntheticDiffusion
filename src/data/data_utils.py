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