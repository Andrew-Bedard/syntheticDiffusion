"""
NN model, training, benchmarking
"""
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import streamlit as st

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


# def train_base_cifar_model(trainloader, testloader, device, epochs):
#     """
#     Trains the base CIFAR-10 model using the full CIFAR-10 dataset and stores the model, its accuracy,
#     and per-class accuracy in the Streamlit session state. Displays the training progress and accuracy
#     in the Streamlit app.
#     """
#     cifar_net = Net()
#     cifar_net.to(device)
#
#     # Display a message while the model is being trained
#     training_message = st.empty()
#     training_message.text("Training base CIFAR-10 model...")
#
#     # Create a progress bar
#     progress_bar = st.progress(0)
#
#     st.session_state.cifar_net = Net()
#     st.session_state.cifar_net.to(device)
#     train(st.session_state.cifar_net, trainloader, device, epochs=epochs, print_every=4000, learning_rate=0.001,
#           momentum=0.9, progress_callback=update_progress, progress_bar=progress_bar)
#
#     # Remove the progress bar after training is complete
#     progress_bar.empty()
#
#     # Clear the message and display the results
#     training_message.empty()
#     cifar_metrics = calculate_metrics(st.session_state.cifar_net, testloader, device)
#     st.session_state.cifar_net_accuracy = cifar_metrics['accuracy']
#     st.session_state.cifar_net_per_class_accuracy = cifar_metrics['per_class_accuracy']
#
#     # st.write(f"CIFAR-10 model accuracy: {st.session_state.cifar_net_accuracy*100:.2f}%")

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
