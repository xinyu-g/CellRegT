import torch
import json
import argparse
import torch.nn as nn
from os import path
import pandas as pd
import numpy as np
from model.Dataset import *
from torch.utils.data import DataLoader
import random
from model.utils import *
from glob import glob
import yaml
from model.Transformer import *
from model.Transformer_SSL import *


class RegisterHooks():
    """ Extract activations (feature maps) from the transformer layers."""
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.hook_b = module.register_full_backward_hook(self.hook_bw)
        
        self.activations = None
        self.gradients = None

    def hook_fn(self, module, input, output):
        self.activations = output # Save activations (output of the layer)
        # self.features = ((output.cpu()).data).numpy()

    def hook_bw(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.hook.remove()
        self.hook_b.remove()


if __name__ == "__main__":
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    input_train = path.realpath(config["LMMPaths"]["output_expression_train"])
    input_test = path.realpath(config["LMMPaths"]["output_expression_test"])
    input_ranked_genes = path.realpath(config["LMMPaths"]["output_ranked_genes_file"])
    n_genes = config['data_processing']['top_k']
    label_col = config["dataset"]["label_col"]
    supervised = config['TransformerModel']['supervised']
    cluster_map_input = config["LMMPaths"]["cluster_map_output"]
    b_size = config['training']['batch_size']
    num_layers = config['TransformerModel']['num_layers']
    model_input = config['TransformerPaths']['model_output']
    input_rn = config['LMMPaths']['output_rn_file']
    num_layers = config['TransformerModel']['num_layers']
    inference_output = config['inference']['inference_output']
    inference_output_normalized = config['inference']['inference_output_norm']

    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu" if not config["dataset"]["device"] else config["dataset"]["device"]
        device = torch.device(dev)
    
    data = pd.read_csv(input_test, index_col=[0])

    f = open(input_ranked_genes, 'r')
    ranked_genes = json.load(f)
    f.close()

    genes = ranked_genes[:n_genes]
    valid = data[genes + [label_col]]

    if check_is_csv(input_rn):
        rn = pd.read_csv(input_rn, index_col=[0])
        rn = rn.to_numpy()
    else:
        rn = np.load(input_rn)


    RN = torch.from_numpy(rn.astype(np.float32))

    RN_ = RN.clone()
    assert RN_.shape[0] == n_genes


    if supervised:
        cluster_map = open(cluster_map_input, 'r')
        word2idx = json.load(cluster_map)
        cluster_map.close()

        word2idx = convert_keys_to_type(word2idx, type(data[label_col].values[0]))

        valid_dataset = MyDataset(valid, word2idx, device=device)
        valid_loader = DataLoader(valid_dataset, b_size, drop_last=True, shuffle=False)
        model = make_classification_model(len(word2idx), RN_, d_model=n_genes, N2=num_layers)
        state_dict = torch.load(model_input, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        num_classes = len(word2idx)
        class_activations = {k: [] for k in range(num_classes)}  # To store activations
        normal_activations = {k: [] for k in range(num_classes)} 

        hooks = RegisterHooks(model.encoder.encoder.layers[0].RN)

        for batch in valid_loader:
            inputs, labels = batch  # Get inputs and labels from the batch

            # Forward pass
            pred, _ = model(inputs)
            
            class_idx = pred.argmax(dim=1)
            for k in range(len(class_idx)):
                model.zero_grad()
                
                pred[k, class_idx[k]].backward(retain_graph=True)

                gradients = hooks.gradients
                activations = hooks.activations

                weighted_activations = activations * gradients
                
                weighted_activations = torch.relu(weighted_activations)

                heatmap = weighted_activations

                class_activations[class_idx[k].item()].append(heatmap)

                if torch.max(heatmap) != 0:
                    heatmap /= torch.max(heatmap)

                normal_activations[class_idx[k].item()].append(heatmap)
            torch.cuda.empty_cache()
                
        avg_activations = {}
        normal_avg_activations = {}

        for k in class_activations.keys():
            avg_activations[k] = torch.mean(torch.stack(class_activations[k]), dim=0) if class_activations[k] else torch.zeros(n_genes,n_genes)
            normal_avg_activations[k] = torch.mean(torch.stack(normal_activations[k]), dim=0) if normal_activations[k] else torch.zeros(n_genes,n_genes)

        torch.save(avg_activations, inference_output)
        torch.save(normal_avg_activations, inference_output_normalized)


    else:
        valid_dataset = ContrastiveDataset(torch.from_numpy(data.to_numpy()), device=device)
        valid_loader = DataLoader(valid_dataset, b_size, drop_last=True, shuffle=False)

        model = make_self_supervised_model(RN_, d_model=n_genes, N2=num_layers)
        state_dict = torch.load(model_input, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        assert np.allclose(
            state_dict['encoder.layers.0.RN.RN'].detach().cpu().numpy(),
            model.encoder.layers[0].RN.RN.detach().cpu().numpy(),
            atol=1e-6  # Adjust tolerance as needed (1e-6 is common)
        ), "The arrays are not close enough!"

        model.eval()
        activations_list = []
        normal_activations = []
        hooks = RegisterHooks(model.encoder.layers[0].RN)

        for batch in valid_loader:
            inputs, labels = batch  # Get inputs and labels from the batch

            # Forward pass
            embeddings = model(inputs, inputs)
            
            for k in range(b_size):
                model.zero_grad()
                
                embeddings[k].mean().backward(retain_graph=True)

                gradients = hooks.gradients
                activations = hooks.activations

                weighted_activations = activations * gradients
                weighted_activations = torch.relu(weighted_activations)
                heatmap = weighted_activations

                activations_list.append(heatmap)

                if torch.max(heatmap) != 0:
                    heatmap /= torch.max(heatmap)

                normal_activations.append(heatmap)
            torch.cuda.empty_cache()
                
        avg_activations = torch.mean(torch.stack(activations_list), dim=0)
        normal_avg_activations = torch.mean(torch.stack(normal_activations), dim=0)

        torch.save(avg_activations, inference_output)
        torch.save(normal_avg_activations, inference_output_normalized)


    
    

