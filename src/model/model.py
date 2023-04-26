"""
NN model, training, benchmarking
"""
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import streamlit as st
from src.data.data_utils import CustomDataset
import random

# Define a callback function to update the progress bar for model training
def update_progress(progress_bar, progress):
    progress_bar.progress(progress)

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
                progress_callback(progress_bar,
                                  (epoch * len(model_trainloader) + i) / (epochs * len(model_trainloader)))

    print('Finished Training')


def train_and_display(model_name, trainloader, testloader, device, epochs):
    """
    Trains a model using the given trainloader and stores the model and its accuracy in the Streamlit session state.
    Displays the training progress and accuracy in the Streamlit app.

    Args:
        model_name (str): A string representing the name of the model.
        trainloader: The training data loader.
        testloader: The test data loader.
        device: The device to train the model on (e.g., "cuda" or "cpu").
        session_key (str): The key to store the model in the Streamlit session state.
        epochs (int): The number of epochs for training the model.
    """
    model = Net()
    model.to(device)

    # Display a message while the model is being trained
    training_message = st.empty()
    training_message.text(f"Training {model_name}...")

    # Create a progress bar
    progress_bar = st.progress(0)

    # Train the model using the trainloader
    train(model, trainloader, device, epochs=epochs, print_every=4000, learning_rate=0.001,
          momentum=0.9, progress_callback=update_progress, progress_bar=progress_bar)

    # Remove the progress bar after training is complete
    progress_bar.empty()

    # Clear the message and display the results
    training_message.empty()

    # Calculate metrics and store them in the session state
    metrics = calculate_metrics(model, testloader, device)
    st.session_state[model_name] = {
        'model': model,
        'metrics': metrics
    }


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

def train_cifar_model_with_cats(percentage, trainset, trainloader, device, epochs=30, custom_images=None):
    """
    Trains a CIFAR-10 classifier model with a specified percentage of cat images in the training dataset,
    either using original cat images or custom synthetic cat images.

    This function creates a new custom dataset with the specified percentage of cat images, then trains
    a new model using the modified dataset. If custom_images is provided, it will use the synthetic cat
    images instead of the original cat images from the trainset.

    Args:
        percentage (float): The percentage of cat images to include in the training dataset, in the range [0, 100].
        trainset (torch.utils.data.Dataset): The original CIFAR-10 training dataset.
        trainloader (torch.utils.data.DataLoader): The original DataLoader for the CIFAR-10 training dataset.
        device (torch.device): The device to use for training the model (e.g., 'cpu' or 'cuda').
        custom_images (torch.tensor, optional): A tensor containing custom synthetic cat images. If provided,
                                                the function will use these images instead of the original cat
                                                images from the trainset.

    Returns:
        model (Net): The trained CIFAR-10 classifier model.
    """
    # Find the indices of the cat images in the trainset
    cat_indices = [i for i, (image, label) in enumerate(trainset) if
                   label == 3]  # Assuming cat class has a label of 3

    # Find the indices of the non-cat images in the trainset
    non_cat_indices = [i for i in range(len(trainset)) if i not in cat_indices]

    # Extract non-cat images and labels
    non_cat_images = [trainset[i][0] for i in non_cat_indices]
    non_cat_labels = [trainset[i][1] for i in non_cat_indices]

    # Calculate the number of cat images to include
    num_cat_images = int(len(cat_indices) * (percentage / 100))

    if num_cat_images > 0:
        if custom_images is None:
            # Randomly choose the cat images to include
            chosen_cat_indices = random.sample(cat_indices, num_cat_images)
            chosen_cat_images = [trainset[i][0] for i in chosen_cat_indices]
            chosen_cat_labels = [trainset[i][1] for i in chosen_cat_indices]

            # Convert chosen_cat_images to a tensor
            chosen_cat_images = torch.stack(chosen_cat_images)
        else:
            # Create custom labels (all cat obviously)
            custom_labels = len(custom_images) * [3]

            # Randomly choose the synthetic cat images to include
            indices = torch.randperm(len(custom_images))[:num_cat_images]
            chosen_cat_images = custom_images[indices]
            chosen_cat_labels = [custom_labels[i] for i in indices.tolist()]

        # Stack non-cat images and chosen cat images
        non_cat_images = torch.stack(non_cat_images)

        # Concatenate non-cat images and chosen cat images
        new_trainset_images = torch.cat((non_cat_images, chosen_cat_images), dim=0)

        # Combine the non-cat labels with the chosen cat labels
        new_trainset_labels = non_cat_labels + chosen_cat_labels
    else:
        new_trainset_images = torch.stack(non_cat_images)
        new_trainset_labels = non_cat_labels

    # Create a new custom dataset with the specified percentage of cat images
    new_trainset = CustomDataset(new_trainset_images, new_trainset_labels)

    # Create a new DataLoader with the modified trainset
    new_trainloader = torch.utils.data.DataLoader(new_trainset, batch_size=trainloader.batch_size, shuffle=True,
                                                  num_workers=trainloader.num_workers)

    # Train the model using the modified trainloader
    model = Net()
    model.to(device)
    train(model, new_trainloader, device, epochs=epochs, print_every=4000, learning_rate=0.001, momentum=0.9)

    return model

def train_and_evaluate_models(percentages, num_trials, trainset, trainloader, testloader, device,
                              custom_images=None):
    """
    Trains and evaluates CIFAR-10 classifier models with different percentages of cat images in the training dataset,
    either using original cat images or custom synthetic cat images.

    This function trains multiple models for each percentage of cat images specified in the 'percentages' list.
    Each model is trained 'num_trials' times and evaluated on the test dataset. If custom_images is provided,
    it will use the synthetic cat images instead of the original cat images from the trainset. The evaluation
    metrics for each model are stored in a dictionary keyed by the percentage of cat images.

    Args:
        percentages (list): A list of percentages of cat images to include in the training dataset, each in the range [0, 100].
        num_trials (int): The number of times each model should be trained and evaluated.
        trainset (torch.utils.data.Dataset): The original CIFAR-10 training dataset.
        trainloader (torch.utils.data.DataLoader): The original DataLoader for the CIFAR-10 training dataset.
        testloader (torch.utils.data.DataLoader): The DataLoader for the CIFAR-10 test dataset.
        device (torch.device): The device to use for training and evaluating the models (e.g., 'cpu' or 'cuda').
        custom_images (torch.tensor, optional): A tensor containing custom synthetic cat images. If provided,
                                                the function will use these images instead of the original cat
                                                images from the trainset.

    Returns:
        all_metrics (dict): A dictionary containing the evaluation metrics for each model, keyed by the percentage
                            of cat images in the training dataset. Each value in the dictionary is a list of metrics
                            dictionaries, one for each trial.
    """
    all_metrics = {p: [] for p in percentages}

    for p in percentages:
        print(p)
        for _ in range(num_trials):
            if custom_images is None:
                model = train_cifar_model_with_cats(p, trainset, trainloader, device)
            else:
                model = train_cifar_model_with_cats(p, trainset, trainloader, device, custom_images=custom_images)

            metrics = calculate_metrics(model, testloader, device)
            all_metrics[p].append(metrics)

    return all_metrics
