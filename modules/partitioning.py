import numpy as np

# Partitioning functions to divide/distribute 'local' datasets
def iid_partition(dataset, n_clients):
    """
    Creating an I.I.D. parition of data to then distribute between clients (i.e. simulating local datasets)

    params:
    - dataset (torch.utils.Dataset):   dataset instance
    - n_clients (int):                 number of Clients between which to split the data

    return: Dict of image indexes for each client
    """
    num_items_per_client = int(len(dataset)/n_clients)
    client_dict = {}
    image_idxs = np.arange(len(dataset))

    # assigning different images to each client
    for cl in range(n_clients):
        client_dict[cl] = set(np.random.choice(image_idxs, 
                                              num_items_per_client, 
                                              replace=False))
        image_idxs = list(set(image_idxs) - client_dict[cl])
    return client_dict


def non_iid_partition(dataset, n_clients, num_shards_per_client):
    """
    Creating a NON-I.I.D paritition of data over clients 
    
    params:
    - dataset (torch.utils.Dataset):  dataset from source
    - n_clients (int):                number of Clients between which to split the data
    - num_shards_per_client (int):    Number of shards to assign each client

    returns: Dict of image indexes for each client
    """

    client_dict  = {i: np.array([], dtype='int64') for i in range(n_clients)}
    total_shards = n_clients * num_shards_per_client
    shards_size  = int(len(dataset)/total_shards)
    shard_idxs   = np.arange(total_shards)
    idxs         = np.arange(len(dataset))
    data_labels  = dataset.targets # .numpy()

    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1,:].argsort()]
    idxs = label_idxs[0,:]

    # assign num_shards_per_client to each client
    for i in range(n_clients):
        i_set = set(np.random.choice(shard_idxs,
                                     num_shards_per_client, 
                                     replace=False))
        shard_idxs = list(set(shard_idxs) - i_set)

        for el in i_set:
            client_dict[i] = np.concatenate(
                (client_dict[i], idxs[el*shards_size:(el+1)*shards_size]), axis=0)
  
    return client_dict