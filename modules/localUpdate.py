import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset




class CustomDataset(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
    

class ClientUpdate(object):
    """ 
    Class for local updates before sending the parameters to the central server
    """
    def __init__(self, algo, model, dataset, local_batchSize, optim, learning_rate, gamma, criterion, local_epochs, idxs, dev):

        self.algo           = algo
        self.model          = model        
        self.learning_rate  = learning_rate
        self.local_epochs   = local_epochs
        self.local_bs       = local_batchSize
        self.optimizer      = optim
        self.criterion      = criterion
        self.gamma          = gamma        
        self.train_loader   = DataLoader(CustomDataset(dataset, idxs), 
                                       batch_size=len(idxs), 
                                       shuffle=True) if local_batchSize == "all" else DataLoader(CustomDataset(dataset, idxs), 
                                                                                                 batch_size=self.local_bs, 
                                                                                                 shuffle=True)
        self.device = dev

    def coupling_loss(self, server_model):
        """
        Adjustment term of the Loss: computes distance of client's parameters from the central server's
        """

        dist = 0.0
        for wr, wc in zip(self.model.parameters(), server_model.parameters()):
            dist += F.mse_loss(wr, wc, reduction = "sum")

        return self.gamma*dist
    
    def train(self, server_model):

        self.model.to(self.device)
        self.model.train()

        # optimizer to update parameters
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        else: print("not implemented optimizer")

        e_loss = []
        for epoch in range(self.local_epochs):

            train_loss = 0.0
            for data, labels in self.train_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                optimizer.zero_grad()                     # Initializing/clearing the gradients to zero                
                output = self.model(data)                 # Forward propagation (forward pass on the nn and get predictions)                
                loss = self.criterion(output, labels)     # Error evaluation, calculating the loss

                if self.algo == "rsgd":
                    loss += self.coupling_loss(server_model)  # adjusting the loss with the distance from the central model weights
                
                loss.backward()                           # Bacward propagation, (backwards pass on the nn)                
                optimizer.step()                          # Optimization step, updating parameters                
                train_loss += loss.item()                 # update training loss

            # average losses
            train_loss /= len(self.train_loader.dataset)
            e_loss.append(train_loss)

        total_loss = sum(e_loss)/len(e_loss)

        return self.model.state_dict(), total_loss

    
class ClientEval(object):
    """ 
    Class for local updates before sending the parameters to the central server
    """
    def __init__(self, algo, model, dataset, local_batchSize, optim, learning_rate, gamma, criterion, local_epochs, idxs, dev):

        self.algo           = algo
        self.model          = model        
        self.local_epochs   = 1
        self.local_bs       = local_batchSize
        self.criterion      = criterion
        self.gamma          = gamma        
        self.train_loader   = DataLoader(CustomDataset(dataset, idxs), 
                                       batch_size=len(idxs), 
                                       shuffle=True) if local_batchSize == "all" else DataLoader(CustomDataset(dataset, idxs), 
                                                                                                 batch_size=self.local_bs, 
                                                                                                 shuffle=True)
        self.device = dev

    def coupling_loss(self, server_model):
        """
        Adjustment term of the Loss: computes distance of client's parameters from the central server's
        """

        dist = 0.0
        for wr, wc in zip(self.model.parameters(), server_model.parameters()):
            dist += F.mse_loss(wr, wc, reduction = "sum")

        return self.gamma*dist
    
    def eval(self, server_model):

        self.model.to(self.device)
        self.model.eval()

        e_loss = []

        with torch.no_grad():
            for epoch in range(self.local_epochs):
                t_loss = 0.0
                
                for data, labels in self.train_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                
                    output = self.model(data)                 # Forward propagation (forward pass on the nn and get predictions)                
                    loss = self.criterion(output, labels)     # Error evaluation, calculating the loss

                    if self.algo == "rsgd":
                        loss += self.coupling_loss(server_model)  # adjusting the loss with the distance from the central model weights
                                   
                    t_loss += loss.item()                 # update training loss

                # average losses
                t_loss /= len(self.train_loader.dataset)
                e_loss.append(t_loss)

            total_loss = sum(e_loss)/len(e_loss)

        return total_loss      
