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
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    

def train(net, trainloader, device, epochs=4, print_every=2000, learning_rate=0.001, momentum=0.9):
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
    """
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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

    print('Finished Training')

def calculate_accuracy(model, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def calculate_per_class_accuracy(model, dataloader, device, num_classes):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    per_class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(num_classes)]
    return per_class_accuracy

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
    return(image)

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
    
    return(transform(cfar_img))

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


# Define some global variables

synthetic_imgDir = "D:\\Projects\\synthetic_diffusion\\data\\cats"

# These are the classes in the cifar-10 dataset (in proper order)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

batch_size = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # If gpu exists use cuda

# Load and transform synthetic images

custom_images = load_and_transform_images(synthetic_imgDir)
custom_labels = len(custom_images)*[3]  # Your assigned labels(cat)

# Cache the dataset loading
@st.cache_resource
def load_cifar10():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainset, trainloader, testset, testloader


trainset, trainloader, testset, testloader = load_cifar10()

# ... (Load and transform synthetic images)
custom_images = load_and_transform_images(synthetic_imgDir)
custom_labels = len(custom_images)*[3]  # Your assigned labels(cat)

# Define slider for selecting number of custom images to add
num_custom_images = st.slider('Select number of custom images to add to dataset', 0, len(custom_images), step=10)

# Normalize CIFAR-10 images
cfar_images = [cfar_transform(cfar_image) for cfar_image in trainset.data]
cfar_images = torch.stack(cfar_images)

# Concatenate our synthetic images with CIFAR-10 images
custom_train_images = torch.cat((cfar_images, custom_images[:num_custom_images]), dim=0)
custom_train_labels = trainset.targets + custom_labels[:num_custom_images]

# Create a custom dataloader for our new combined dataset
custom_trainset = CustomDataset(custom_train_images, custom_train_labels)
custom_trainloader = torch.utils.data.DataLoader(custom_trainset, batch_size=batch_size,
                                                 shuffle=True, num_workers=0)

# Initialize variables to store the previous custom model's accuracies in the session state
if 'prev_custom_net_accuracy' not in st.session_state:
    st.session_state.prev_custom_net_accuracy = None

if 'prev_custom_net_per_class_accuracy' not in st.session_state:
    st.session_state.prev_custom_net_per_class_accuracy = None

# Initialize variables to store the trained cifar_net in the session state
if 'cifar_net' not in st.session_state:
    st.session_state.cifar_net = None

# Create a button for triggering the training process
if st.button('Train the models'):

    # Display a message while the model is being trained
    training_message = st.empty()
    training_message.text("Training models...")

    # Create two instances of your model
    custom_net = Net()
    custom_net.to(device)

    cifar_net = Net()
    cifar_net.to(device)

    # Train the networks using the custom and CIFAR-10 dataloaders
    train(custom_net, custom_trainloader, device, epochs=5, print_every=4000, learning_rate=0.001, momentum=0.9)

    # Train the cifar_net only if it hasn't been trained yet
    if st.session_state.cifar_net is None:
        st.session_state.cifar_net = Net()
        st.session_state.cifar_net.to(device)
        train(st.session_state.cifar_net, trainloader, device, epochs=5, print_every=4000, learning_rate=0.001, momentum=0.9)


    # Clear the message and display the results
    training_message.empty()

    # Calculate overall accuracy
    custom_net_accuracy = calculate_accuracy(custom_net, testloader, device)
    cifar_net_accuracy = calculate_accuracy(st.session_state.cifar_net, testloader, device)

    # Calculate per-class accuracy
    num_classes = 10
    custom_net_per_class_accuracy = calculate_per_class_accuracy(custom_net, testloader, device, num_classes)
    cifar_net_per_class_accuracy = calculate_per_class_accuracy(st.session_state.cifar_net, testloader, device, num_classes)

    # Display overall accuracy
    st.subheader("Overall Model Accuracy")
    st.write(f"Custom model accuracy: {custom_net_accuracy:.2f}%")
    st.write(f"CIFAR-10 model accuracy: {cifar_net_accuracy:.2f}%")

    # Plot per-class accuracy
    st.subheader("Per-class Model Accuracy")
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, custom_net_per_class_accuracy, width, label='Custom Model')
    rects2 = ax.bar(x + width/2, cifar_net_per_class_accuracy, width, label='CIFAR-10 Model')

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
        st.write(f"Previous custom model accuracy: {st.session_state.prev_custom_net_accuracy:.2f}%")
        st.write(f"New custom model accuracy: {custom_net_accuracy:.2f}%")

        # Plot per-class accuracy comparison
        fig2, ax2 = plt.subplots()
        rects3 = ax2.bar(x - width/2, st.session_state.prev_custom_net_per_class_accuracy, width, label='Previous Custom Model')
        rects4 = ax2.bar(x + width/2, custom_net_per_class_accuracy, width, label='New Custom Model')

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
    st.session_state.prev_custom_net_accuracy = custom_net_accuracy
    st.session_state.prev_custom_net_per_class_accuracy = custom_net_per_class_accuracy


# TODO: Create new images
# TODO: Add some image interactivity
# TODO: Save synthetic images in cifar format to allow for creation of git repo
# TODO: more optimization of NN architecture?
# TODO: See how easy it would be to save the cifar NN to avoid retraining it all the time
# TODO: split this project up into a few files and use imports to clean it up
