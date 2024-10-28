import torch
import json
import argparse
import torch.nn as nn
from os import path
import pandas as pd
import numpy as np
import yaml
from model.utils import *
from model.Dataset import *
from torch.utils.data import DataLoader
from model.Transformer import *
from model.Transformer_SSL import *



if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    input_train = path.realpath(config["LMMPaths"]["output_expression_train"])
    input_test = path.realpath(config["LMMPaths"]["output_expression_test"])
    input_ranked_genes = path.realpath(config["LMMPaths"]["output_ranked_genes_file"])
    input_rn = config['LMMPaths']['output_rn_file']
    dname = config["dataset"]["name"]
    cluster_map_input = config["LMMPaths"]["cluster_map_output"]
    label_col = config["dataset"]["label_col"]
    supervised = config['TransformerModel']['supervised']
    model_output = config['TransformerPaths']['model_output']
    n_genes = config['data_processing']['top_k']
    dname = config['dataset']['name']
    lr = config['training']['model_lr']
    num_epochs = config['training']['epochs']
    b_size = config['training']['batch_size']
    num_layers = config['TransformerModel']['num_layers']
    latent_dim = config['TransformerModel']['latent_dim']

    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu" if not config["dataset"]["device"] else config["dataset"]["device"]
        device = torch.device(dev)
    

    train_data = pd.read_csv(input_train, index_col=[0])
    if check_is_csv(input_rn):
        rn = pd.read_csv(input_rn, index_col=[0])
        rn = rn.to_numpy()
    else:
        rn = np.load(input_rn)


    RN = torch.from_numpy(rn.astype(np.float32))
    # number of clusters
    genes_file = open(input_ranked_genes, 'r')
    ranked_genes = json.load(genes_file)
    genes_file.close()

    

    RN_ = RN.clone()
    assert RN_.shape[0] == n_genes


    if supervised:
        cluster_map = open(cluster_map_input, 'r')
        word2idx = json.load(cluster_map)
        cluster_map.close()

        word2idx = convert_keys_to_type(word2idx, type(train_data[label_col].values[0]))
        k = len(word2idx)


        test_data = pd.read_csv(input_test, index_col=[0])
        train_data = train_data[ranked_genes[:n_genes] + [label_col]]
        test_data = test_data[ranked_genes[:n_genes] + [label_col]]
        train_dataset = MyDataset(train_data, word2idx, device=device)
        test_dataset = MyDataset(test_data, word2idx, device=device)
        model = make_classification_model(len(word2idx), RN_, d_model=n_genes, N2=num_layers)
        model.to(device)
        
        is_rn = False if not num_layers else True
        run_model(model, RN, num_epochs, b_size, train_dataset, test_dataset, n_genes, lr, is_rn, model_output)

    else:
        train_data = train_data[ranked_genes[:n_genes]]
        train_dataset = ContrastiveDataset(torch.from_numpy(train_data.to_numpy()), device=device)

        model = make_self_supervised_model(RN_, d_model=n_genes, N2=num_layers)
        model.to(device)

        train_contrastive_model(model, train_dataset, n_epochs=num_epochs, batch_size=b_size, lr=lr, save_path=model_output)
    