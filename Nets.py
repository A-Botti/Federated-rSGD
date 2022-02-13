import torch
import torch.nn as nn
import torch.nn.functional as F


######################################
######          LeNet5         #######
######################################

class mnist_LeNet5(nn.Module):
    """
    CNN with two 5x5 convolution layers (the first with 6 filters, the second with 16,
    each followed by 2x2 max pooling), 3 fully connected layers with 120, 84 and 10 units and 
    ReLu activations
    """
    def __init__(self):
        super().__init__()

        # convolutions     
        # MNIST is 28x28 input size 
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 5) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)

        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #10 classes for the MNIST
        
    def forward(self, x):
      
        # activations and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten to single fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 

        return x
    
class cifar_LeNet5(nn.Module):
    """
    LeNet similar to above, but for 32x32 images and 3 color channels
    """
    def __init__(self):
        super().__init__()

        # convolutions     
        # CIFAR10 has 32x32 images with 3 channels RGB
        self.conv1 = nn.Conv2d(3, 6, kernel_size = 5) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5)

        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #10 classes for the MNIST
        
    def forward(self, x):
      
        # activations and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten to single fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 

        return x
    
    
######################################
######           CNN           #######
######################################

class mnist_CNN(nn.Module):
    """
    CNN with two 5x5 convolution layers (the first with 32 filters, the second with 64,
    each followed by 2x2 max pooling), a fully connected layers with 512 units and 
    ReLu activations with a final softmax output layer
    """
    def __init__(self):
        super().__init__()

        # convolutions     
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)

        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10) # 10 classes

    def forward(self, x):
      
        # activations and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten to single fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    
class cifar_CNN(nn.Module):
    """
    Similar to above but for CIFAR10 images
    """
    def __init__(self):
        super().__init__()

        # convolutions     
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5)

        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # fully connected layers
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, 10) # last before softmax

        
    def forward(self, x):
      
        # activations and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten to single fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x