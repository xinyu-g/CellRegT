import torch
import json
import argparse
import torch.nn as nn
from os import path
import pandas as pd
import numpy as np
from model.utils import *
from model.Dataset import *
from torch.utils.data import DataLoader
from model.LMM import *
import yaml




if __name__ == "__main__":

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    
    input_train = path.realpath(config["LMMPaths"]["input_scRNAseq_train"])
    input_test = path.realpath(config["LMMPaths"]["input_scRNAseq_test"])
    output_ranked_genes = path.realpath(config["LMMPaths"]["output_ranked_genes_file"])
    output_train = path.realpath(config["LMMPaths"]["output_expression_train"])
    output_test = path.realpath(config["LMMPaths"]["output_expression_test"])
    output_rn = config['LMMPaths']['output_rn_file']
    dname = config["dataset"]["name"]
    cluster_map_output = config["LMMPaths"]["cluster_map_output"]
    label_col = config["dataset"]["label_col"]
    supervised = config['LMMmodel']['supervised']
    model_output = config['LMMPaths']['model_output']
    latent_dim = config['LMMmodel']['latent_dim']
    num_random_effects = config['LMMmodel']['num_random_effects']
    n_genes = config['data_processing']['top_features_to_save']
    top_k = config['data_processing']['top_k']
    get_RN = config['data_processing']['get_RN']
    dname = config['dataset']['name']
    lr = config['training']['model_lr']
    num_epochs = config['training']['epochs']
    b_size = config['training']['batch_size']


    

    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu" if not config["dataset"]["device"] else config["dataset"]["device"]
        device = torch.device(dev)
    

    train_data = pd.read_csv(input_train, index_col=[0])
    

    num_samples = len(train_data)


    random_effect_data = torch.randn(num_samples, num_random_effects)
    


    

    if supervised:
        clusters = train_data[label_col].unique()
        k = len(clusters)

        word2idx = {p: idx for idx, p in enumerate(clusters)}

        cluster_map = open(cluster_map_output, 'w')
        json.dump({str(p): idx for idx, p in enumerate(clusters)}, cluster_map)
        cluster_map.close()

        labels = train_data[label_col].map(word2idx)
        gene_col = [col for col in train_data.columns if col != label_col]
        
        num_genes = len(gene_col)
        expression_data = torch.from_numpy(train_data[gene_col].to_numpy()).to(torch.float32)
        train_dataset = GeneExpressionDataset(expression_data, random_effect_data, labels.to_numpy())
        data_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        test_data = pd.read_csv(input_test, index_col=[0])
        model = SupervisedLinearMixedModel(input_dim=num_genes, latent_dim=latent_dim, random_effect_dim=num_random_effects, output_dim=k)
        model = train_supervised_model(data_loader, model, lr=lr, num_epochs=num_epochs)
        features = gene_col

    else:
        expression_data = torch.from_numpy(train_data.to_numpy()).to(torch.float32)
        train_dataset = GeneExpressionDataset(expression_data, random_effect_data)
        data_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        num_genes = len(train_data.columns)
        model = SelfSupervisedLinearMixedModel(input_dim=num_genes, latent_dim=latent_dim, random_effect_dim=num_random_effects)
        model = train_self_supervised_model(data_loader, model, lr=lr, num_epochs=num_epochs)
        features = train_data.columns.to_list()

    save_model(model, model_output)


    fixed_coefficients, _ = extract_fixed_importance(model, aggregation_method='sum')

    coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': fixed_coefficients})
    coefficients_df.sort_values(by='Coefficient', ascending=False, inplace=True)
    features = coefficients_df['Feature'].to_list()

    genes_file = open(output_ranked_genes, 'w')
    json.dump(features, genes_file)
    genes_file.close()

    train_data[features[:n_genes]].to_csv( output_train, index=True)

    if supervised:
        train_data[features[:n_genes] + [label_col]].to_csv( output_train, index=True)
        test_data[features[:n_genes] + [label_col]].to_csv( output_test, index=True)


    if get_RN:
        colltri = pd.read_csv(config["LMMPaths"]['db_path'])
        save_network_dorothea(colltri, dname, output_rn, top_k, features)

        # can take a long time due to api stability
        # save_networks_SD(dname, output_rn, top_k, features)


