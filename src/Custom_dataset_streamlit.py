import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from data.data_utils import resize_normalize, cfar_transform, load_and_transform_images, subsample_dataset, CustomDataset
from model.model import Net, train, calculate_accuracy, calculate_per_class_accuracy
from viewer.visualization_utils import imshow


# Define some global variables
torch.manual_seed(0)
synthetic_imgDir = "D:\\Projects\\synthetic_diffusion\\data\\cats"

# These are the classes in the cifar-10 dataset (in proper order)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

batch_size = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # If gpu exists use cuda
######################################################################################################################

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
if st.button('Train models'):

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



