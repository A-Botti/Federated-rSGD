import copy
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils, datasets


def get_dataset(dataset):
    """
    Retrieve the selected dataset train and test set 
    
    Param: dataset (string) : either 'mnist', 'fminst' or 'cifar10'
    returns: train and test splits from source
    """
    
    if dataset == "mnist":

        # Same transform used for train and test
        transform = transforms.Compose([#transforms.Pad(2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.1307,),
                                                             std = (0.3081,))
                                      ])

        # will download dataset if not already downloaded; meanwhile applying the transformations
        trainset = datasets.MNIST('../data/mnist/', train=True, download=True, transform=transform)
        testset = datasets.MNIST('../data/mnist/', train=False, download=True, transform=transform)
        
        
    elif dataset == "fmnist":

        # Same transform used for train and test
        transform = transforms.Compose([#transforms.Pad(2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.1307,),
                                                             std = (0.3081,))
                                      ])

        # will download dataset if not already downloaded; meanwhile applying the transformations
        trainset = datasets.FashionMNIST('../data/fmnist/', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST('../data/fmnist/', train=False, download=True, transform=transform) 


    elif dataset == "cifar10":
        
        # Different transform for train & test, we do some augmentation on training
        transform_train = transforms.Compose([transforms.RandomCrop(size = (32,32), padding = 4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                                                   std = (0.229, 0.224, 0.225))
                                      ])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.485, 0.456, 0.406),
                                                                  std = (0.229, 0.224, 0.225))
                                      ])
        # Download CIFAR10 dateset Training and Test data while applying transforms
        trainset = datasets.CIFAR10('../data/cifar/', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10('../data/cifar/', train=False, download=True, transform=transform_test)
        
    return trainset, testset

def server_aggregate(w):
    """
    Updating the global weights by averaging client's results
    Params:
    - w (list of model.state_dict()): list of the k clients' model parmeters that are sent to the server at each communication round

    Returns: server updated weights
    """
    weights_avg = copy.deepcopy(w[0])
    for k in weights_avg.keys():
        weights_avg[k] = torch.stack([w[i][k] for i in range(len(w))], 0).mean(0)
    
    return weights_avg # round aggregated/averaged parameters


def test_metrics(model, test_ds, criterion, e_batchSize):
    """
    Function to evaluate model's performance accuracy. Will be called at every communication round

    Params:
    - model (nn.Module):       instance of the NN
    - test_ds (nn.Dataset):    Dataset used for evaluation
    - criterion (nn.Loss):     Loss function to evaluate

    returns: test set loss and accuracy
    """

    model.eval()                  # set model in evaluation mode
    loss, correct = 0.0, 0.0      # placeholder

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_loader = DataLoader(test_ds, batch_size=e_batchSize)

    # inference and prediction for each batch
    for data, labels in test_loader:      
        data, labels = data.to(device), labels.to(device)

        output = model(data)
        loss += criterion(output, labels).item()                #sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)               #get predicted class index
        correct += pred.eq(labels.view_as(pred)).sum().item()

    loss /= len(test_loader.dataset)
    accuracy = correct/len(test_loader.dataset)
    
    return loss, accuracy