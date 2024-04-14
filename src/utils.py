from torch.utils.data import random_split
import numpy as np
import requests
import torch


def onehot(idx, length):
    lst = [0 for i in range(length)]
    lst[idx] = 1
    return lst 

def split(ratio, data):
    train_set_size = int(len(data) * ratio)
    valid_set_size = len(data) - train_set_size
    train_set, valid_set = random_split(data, [train_set_size, valid_set_size])
    return train_set, valid_set

def split_df(ratio, data):
    np.random.seed(0)
    # Shuffle indices
    shuffled_indices = np.random.permutation(data.index)
    
    # Calculate split index
    split_index = int(len(data) * ratio)
    
    # Split indices into train and valid
    train_indices = shuffled_indices[:split_index]
    valid_indices = shuffled_indices[split_index:]
    
    # Create train and valid datasets
    train_dataset = data.loc[train_indices]
    valid_dataset = data.loc[valid_indices]
    
    return train_dataset, valid_dataset


def run_basic_model(model,x,y,test_x):
    model.fit(x,y)
    y_pred = model.predict_proba(test_x)

    return y_pred


def get_network_SD(geneList):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "json"
    method = "network"

    request_url = "/".join([string_api_url, output_format, method])


    my_genes = geneList

    params = {

        "identifiers" : "%0d".join(my_genes),
        "species" : 'Mammals', # species NCBI identifier 
        "caller_identity" : "GT" 

    }

    response = requests.post(request_url, data=params)

    return response


def move_layers_to_gpu(layers, num_layers_to_gpu, device):
    for i, layer in enumerate(layers):
        if i < num_layers_to_gpu:
            layer.to(device)


def move_layers_to_cpu(layers, low, high):
    for i, layer in enumerate(layers):
        if low <= i <= high:
            layer.to(torch.device('cpu'))