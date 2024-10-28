from torch.utils.data import random_split
import numpy as np
import requests
import torch
import torch.optim as optim
import colorsys
import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

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
    data = data.reset_index()
    shuffled_indices = np.random.permutation(data.index)
    
    print(shuffled_indices)
    # Calculate split index
    split_index = int(len(data) * ratio)

    print(split_index)
    
    # Split indices into train and valid
    train_indices = shuffled_indices[:split_index]
    valid_indices = shuffled_indices[split_index:]
    print(len(train_indices), len(valid_indices))
    print(type(train_indices), type(valid_indices))

    print(f"Shuffled indices length: {len(shuffled_indices)}")
    print(f"Original data length: {len(data)}")

    
    # Create train and valid datasets
    train_dataset = data.loc[train_indices]
    valid_dataset = data.loc[valid_indices]
    print(len(train_dataset), len(valid_dataset))
    return train_dataset, valid_dataset


def split_df_by_col(ratio, data, column):
    np.random.seed(0)

    data = data.reset_index()
    
    # Get unique values in the specified column
    unique_values = data[column].unique()

    
    train_dataset_list = []
    valid_dataset_list = []
    
    # Loop over each unique value and split the data
    for value in unique_values:
        # Get all rows corresponding to the current unique value
        subset = data[data[column] == value]
        
        # Shuffle the subset
        shuffled_indices = np.random.permutation(subset.index)
        
        # Calculate split index for the subset
        split_index = int(len(subset) * ratio)
        
        # Split into train and valid datasets
        train_indices = shuffled_indices[:split_index]
        valid_indices = shuffled_indices[split_index:]
        
        # Append the subsets to the corresponding lists
        train_dataset_list.append(subset.loc[train_indices])
        valid_dataset_list.append(subset.loc[valid_indices])
    
    # Concatenate all the subsets for both train and valid datasets
    train_dataset = pd.concat(train_dataset_list).sample(frac=1, random_state=0)  # Shuffle after concatenation
    valid_dataset = pd.concat(valid_dataset_list).sample(frac=1, random_state=0)  # Shuffle after concatenation
    
    return train_dataset, valid_dataset


def convert_keys_to_int(d):
    new_dict = {}
    for key, value in d.items():
        try:
            # Attempt to convert the key to an integer
            new_key = int(key)
        except ValueError:
            # If it fails, keep the original string key
            new_key = key
        # Add the new key and value to the new dictionary
        new_dict[new_key] = value
    return new_dict


def convert_keys_to_type(d, key_type):
    new_dict = {}
    for key, value in d.items():
        try:
            # Attempt to convert the key to the specified type
            new_key = key_type(key)
        except (ValueError, TypeError):
            # If it fails, keep the original key
            new_key = key
        # Add the new key and value to the new dictionary
        new_dict[new_key] = value
    return new_dict

def run_basic_model(model,x,y,test_x):
    model.fit(x,y)
    y_pred = model.predict_proba(test_x)

    return y_pred


def get_network_SD(geneList, update_params=None):
    # string_api_url = "https://version-11-5.string-db.org/api"
    string_api_url = 'https://string-db.org/api'
    output_format = "json"
    method = "network"

    ##
    ## Construct URL
    ##

    request_url = "/".join([string_api_url, output_format, method])

    # print(request_url)

    my_genes = geneList

    params = {

        "identifiers" : "%0d".join(my_genes), # your protein
        "species" : 'Mammals', # species NCBI identifier 
        "caller_identity" : "GT" # your app name
    }

    if update_params:
        params.update(update_params)

    # print(params)

    # add = ''

    # for k, v in params.items():
        
    #     add += f'{k}={v}&'

    # request_url += '?' + add[:-1]
    # ##
    # ## Call STRING
    # ##

    # print(request_url)
    # response = requests.post(request_url)

    response = requests.post(request_url, data=params, timeout=6000)

    print(response.url)

    return response


def save_network_dorothea(dorothea, dname, path, n, features):
    genes = features[:n]
    dorothea = dorothea[(dorothea['source'].isin(genes)) & (dorothea['target'].isin(genes))]
    interactions = dorothea.drop_duplicates()
    genes_A = interactions['source'].unique().tolist()
    genes_B = interactions['target'].unique().tolist()
    genes_all = list(set(genes_A + genes_B + features[:n]))
    network  = pd.DataFrame(0.0, columns=genes_all, index=genes_all)
    interaction_dict = dict(zip(zip(interactions['source'], interactions['target']), interactions['mor'])) #mode of regulation

    for k, v in interaction_dict.items():
        gene_A = k[0]
        gene_B = k[1]
        network.loc[gene_A, gene_B] = float(v)

    # interactions.to_csv(f'{path}/{dname}_collectTri_interactions_{n}.csv')
    network.to_csv(f'{path}')


def check_is_csv(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension == '.csv':
        return True
    elif file_extension in ['.npz', '.npy']:
        return False
    else:
        raise ValueError("The file is neither a CSV nor a Numpy (.npz/.npy) file.")


def save_networks_SD(dname, path, n, features):
    
    for _ in range(1000):
        try:
            
            # params = {"species" : 9606}
            response = get_network_SD(features[:n])
            # response = get_network_SD(features[:500])

            # print(response)

            interactions = pd.DataFrame.from_records(response.json())
            interactions = interactions[interactions['escore'] + interactions['dscore'] > 0]
            interactions = interactions.drop_duplicates()

            genes_A = interactions['preferredName_A'].unique().tolist()
            genes_B = interactions['preferredName_B'].unique().tolist()
            genes_all = list(set(genes_A + genes_B + features[:n]))

            network  = pd.DataFrame(0.0, columns=genes_all, index=genes_all)
            interaction_dict = dict(zip(zip(interactions['preferredName_A'], interactions['preferredName_B']), interactions['score']))

            for k, v in interaction_dict.items():
                gene_A = k[0]
                gene_B = k[1]
                network.loc[gene_A, gene_B] = v
                network.loc[gene_B, gene_A] = v
            # interactions.to_csv(f'{path}/{dname}_SD_interactions_{n}.csv')
            if check_is_csv(f'{path}'):
                network.to_csv(f'{path}')
            else:
                np.savez_compressed("large_arrays.npz", array1=network.to_numpy()) 

            break

        except Exception as e:
            print('exception:', e)


def move_layers_to_gpu(layers, num_layers_to_gpu, device):
    for i, layer in enumerate(layers):
        if i < num_layers_to_gpu:
            layer.to(device)


def move_layers_to_cpu(layers, low, high):
    for i, layer in enumerate(layers):
        if low <= i <= high:
            layer.to(torch.device('cpu'))



def remove_and_insert(lst, values, i):
    # Check if the value exists in the list
    for value in values:
        if value in lst:
            # Find the index of the value
            original_index = lst.index(value)
            
            # Remove the value from its original position
            removed_value = lst.pop(original_index)
            
            # Insert the value at the specified index (i)
            lst.insert(i, removed_value)
            
            print(f"Value '{value}' moved from index {original_index} to index {i}")
        else:
            print(f"Value '{value}' not found in the list")


def normalize(data, min_genes_per_cell=2000, min_cells_per_gene=3):
    # Basic filtering: filter out cells with fewer than 200 genes and genes expressed in fewer than 3 cells
    cell_counts = (data > 0).sum(axis=1)
    gene_counts = (data > 0).sum(axis=0)

    filtered_data = data.loc[cell_counts >= min_genes_per_cell, gene_counts >= min_cells_per_gene]

    # Normalize the data (CPM normalization)
    counts_per_cell = filtered_data.sum(axis=1)
    normalized_data = filtered_data.div(counts_per_cell, axis=0) * 1e6

    # Log-transform the data
    log_normalized_data = np.log1p(normalized_data)

    return log_normalized_data


def remove_and_insert(lst, values, i):
    # Check if the value exists in the list
    for value in values:
        if value in lst:
            # Find the index of the value
            original_index = lst.index(value)
            
            # Remove the value from its original position
            removed_value = lst.pop(original_index)
            
            # Insert the value at the specified index (i)
            lst.insert(i, removed_value)
            
            print(f"Value '{value}' moved from index {original_index} to index {i}")
        else:
            print(f"Value '{value}' not found in the list")


def analyze_parameter_influence(model, data_loader, param_name, target_class, learning_rate=0.01, num_epochs=1):
    
    model.train()
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    param_influences = []
    # optimizer = optim.SGD([param for name, param in model.named_parameters() if name == param_name], lr=learning_rate)
    
    for batch in data_loader:
        inputs, labels = batch
        inputs.requires_grad_()
            
        # optimizer.zero_grad()
        
        outputs = model(inputs, labels, None, None)
        outputs = model.generator(outputs)

        # print(outputs.shape)
        grad_output = torch.zeros_like(outputs)
        grad_output[:, target_class] = 1.0  # Set gradient for target class
        
        # print(grad_output)
        # Compute gradients of the target class score w.r.t. the model parameters
        grads = grad(outputs, model.parameters(), grad_outputs=grad_output, create_graph=True, allow_unused=True)
        
        # Find the parameter corresponding to param_name
        param_grad = None
        for name, param in model.named_parameters():
            if name == param_name:
                # Get the gradient corresponding to this parameter
                for g in grads:
                    if g.shape == param.shape:
                        param_grad = g
                        break
                break
        
        if param_grad is None:
            raise ValueError(f"Parameter '{param_name}' not found in model parameters.")
        
        param_influences.append(param_grad.detach().cpu().numpy())
    
    return param_influences


def analyze_parameter_influence_2(model, data_loader, param_name, target_class, learning_rate=0.01, num_epochs=1):
    model.train()
    param_influences = []
    optimizer = optim.SGD([param for name, param in model.named_parameters() if name == param_name], lr=learning_rate)
    
    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs, labels = batch
            inputs.requires_grad_()
            
            optimizer.zero_grad()

            # print(inputs.shape, labels.shape)
            
            outputs, _ = model(inputs, labels, None, None)
            outputs = model.generator(outputs)

            
            # Compute loss based on the target class
            loss = outputs[:, target_class].sum()
            loss.backward(retain_graph=True)
            
            # Update the parameter corresponding to param_name
            optimizer.step()
        
        # Collect the updated gradients after each epoch
        for name, param in model.named_parameters():
            if name == param_name:
                param_influences.append(param.grad.detach().cpu().numpy())
    
    return param_influences


def get_interactions(rn, genes, gene_loc, th=0.001):
    edges = []
    for g1 in genes:
        for g2 in genes:
            if g1 != g2:
                w = rn[gene_loc[g1]][gene_loc[g2]].item()
                if abs(round(w,3)) > th and (sorted([g1,g2]) not in edges):
                    edges.append(sorted([g1,g2]))
    return edges

def get_edges_for_g(rn, genes, gene_loc, g1, th=0.001):
    edges = []
    
    for g2 in genes:
        if g1 != g2:
            w = rn[gene_loc[g1]][gene_loc[g2]].item()
            if abs(round(w,3)) > th and (sorted([g1,g2]) not in edges):
                edges.append(sorted([g1,g2]))
    return edges

def Graph_for_g(rn, genes, gene_loc, g1, th=0.001):
    G = nx.Graph()

    for g in genes:
        G.add_node(g)

    pos = nx.kamada_kawai_layout(G) 

    edges = get_edges_for_g(rn, genes, gene_loc, g1)

    for g1, g2 in edges:
        w = rn[gene_loc[g1]][gene_loc[g2]].item()
        if (abs(round(w,3)) > th):
            # print(g1, g2, w, round(w,3))
            G.add_edge(g1, g2, weight=round(w,3))

    return pos, G

def Graph(rn, genes, gene_loc, th=0.001):
    G = nx.Graph()

    for g in genes:
        G.add_node(g)

    pos = nx.kamada_kawai_layout(G) 

    edges = get_interactions(rn, genes, gene_loc, th)

    for g1, g2 in edges:
        w = rn[gene_loc[g1]][gene_loc[g2]].item()
        if (abs(round(w,3)) > th):
            # print(g1, g2, w, round(w,3))
            G.add_edge(g1, g2, weight=round(w,3))

    return pos, G


# Function to convert hex to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Function to convert RGB to hex
def rgb_to_hex(rgb_color):
    return '#%02x%02x%02x' % rgb_color

# Function to adjust the lightness of an RGB color
def adjust_lightness(rgb_color, factor):
    # Convert RGB to HLS
    h, l, s = colorsys.rgb_to_hls(*[x / 255.0 for x in rgb_color])
    # Adjust lightness
    l = max(0, min(1, l * factor))
    # Convert HLS back to RGB
    return tuple(int(x * 255) for x in colorsys.hls_to_rgb(h, l, s))


def get_hex_variation(base_hex, n):

    # Convert hex to RGB
    base_rgb = hex_to_rgb(base_hex)

    # Generate variations by adjusting lightness
    factors = np.linspace(0.5, 1.5, n)  # Adjust the range as needed
    variations = [adjust_lightness(base_rgb, factor) for factor in factors]

    # Convert variations to hex
    variations_hex = [rgb_to_hex(color) for color in variations]

    # Create a gradient image for visualization
    gradient_image = np.zeros((50, n * 50, 3), dtype=np.uint8)

    for i, color in enumerate(variations):
        gradient_image[:, i * 50:(i + 1) * 50] = color

    # Plot the gradient
    plt.imshow(gradient_image)
    plt.axis('off')
    plt.show()

    return variations_hex


def plot_G_compare(rn_origin, rn, genes, gene_loc, cell, path, pos=None, g1=None, th=0.001, size=800, palette=None, width=6):
    
    pos, G1 = Graph(rn, genes, gene_loc, th)

    _, G2 = Graph(rn_origin, genes, gene_loc, th)

    viridis_palette = sns.color_palette('PuBuGn', n_colors=25)
    viridis_palette = viridis_palette[::-1]


    if palette:
        viridis_palette = palette
    node_sizes = [size + i*100 for i in range(25)]
    node_sizes = node_sizes[::-1]

    # pos, G1 = Graph_for_g(rn, genes, gene_loc, g1)

    # _, G2 = Graph_for_g(rn_origin, genes, gene_loc, g1)

    plt.figure(figsize=(15,15))

    ax = plt.subplot(1,1,1)
    # for (u, v, d) in G1.edges(data=True):
    #     print(u, v, d, d['weight'])
    elarge = [(u, v) for (u, v, d) in G1.edges(data=True) if d["weight"] > 0]
    esmall = [(u, v) for (u, v, d) in G1.edges(data=True) if d["weight"] <= 0]

    # pos = nx.kamada_kawai_layout(G1)
    # nx.draw_networkx_nodes(G1, pos, node_size=700)
    l = G1.number_of_nodes()
    
    nx.draw_networkx_nodes(G1, pos, node_size=node_sizes[:l], node_color=viridis_palette[:l])
    nx.draw_networkx_edges(G1, pos, edgelist=elarge, width=width, edge_color="r")
    nx.draw_networkx_edges(
        G1, pos, edgelist=esmall, width=width, alpha=0.5, edge_color="b"
    )

    # node labels
    nx.draw_networkx_labels(G1, pos, font_family="sans-serif", font_size=20, font_color='white', font_weight='bold')
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G1, "weight")
    nx.draw_networkx_edge_labels(G1, pos, edge_labels, label_pos=0.6)
    # ax.text(0.1, 1.05, 'b', transform=ax.transAxes,
    #   fontsize=16, fontweight='bold', va='top', ha='right')

    ax = plt.gca()
    ax.margins(0.08)

    plt.axis("off")
    # plt.title(f'Cell - {cell}_after')
    plt.tight_layout()
    
    plt.savefig(f'{path}/{cell}_network_changes_after.svg')
    # plt.show()
    plt.close()


    plt.figure(figsize=(15,15))

    ax = plt.subplot(1,1,1)
    elarge = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] > 0]
    esmall = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] <= 0]

    # pos = nx.spring_layout(G2) 
    
    # nx.draw_networkx_nodes(G2, pos, node_size=700)
    l = G2.number_of_nodes()
    nx.draw_networkx_nodes(G2, pos, node_size=node_sizes[:l], node_color=viridis_palette[:l])
    nx.draw_networkx_edges(G2, pos, edgelist=elarge, width=width, edge_color="r")
    # nx.draw_networkx_edges(
    #     G2, pos, edgelist=esmall, width=width
    # )
    nx.draw_networkx_edges(
        G2, pos, edgelist=esmall, width=width, alpha=0.5, edge_color="b"
    )

    # node labels
    nx.draw_networkx_labels(G2, pos, font_family="sans-serif", font_size=20, font_color='white', font_weight='bold')
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G2, "weight")
    # nx.draw_networkx_edges(G, pos, ax=ax, edge_labels=edge_labels)
    nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels, label_pos=0.6)

    ax = plt.gca()
    ax.margins(0.08)

    plt.axis("off")
    # plt.title(f'Cell - {cell}_before')
    plt.tight_layout()
    
    
    plt.savefig(f'{path}/{cell}_network_before.svg')
    # plt.show()
    plt.close()


    print(G1.number_of_edges(), G2.number_of_edges())

    return G1, G2

def plot_G_compare_for_g(rn_origin, rn, genes, gene_loc, cell, path, pos=None, g1=None, th=0.001, n_colors=25, size=800, palette=None, width=6):
    

    viridis_palette = sns.color_palette('PuBuGn', n_colors=n_colors)
    viridis_palette = viridis_palette[::-1]
    if palette:
        viridis_palette = palette
    node_sizes = [size + i*100 for i in range(n_colors)]
    node_sizes = node_sizes[::-1] 

    pos, G1 = Graph_for_g(rn, genes, gene_loc, g1)

    _, G2 = Graph_for_g(rn_origin, genes, gene_loc, g1)

    plt.figure(figsize=(15,15))

    ax = plt.subplot(1,1,1)
    # for (u, v, d) in G1.edges(data=True):
    #     print(u, v, d, d['weight'])
    elarge = [(u, v) for (u, v, d) in G1.edges(data=True) if d["weight"] > 0]
    esmall = [(u, v) for (u, v, d) in G1.edges(data=True) if d["weight"] <= 0]

    # pos = nx.kamada_kawai_layout(G1)
    # nx.draw_networkx_nodes(G1, pos, node_size=700)
    l = G1.number_of_nodes()
    
    nx.draw_networkx_nodes(G1, pos, node_size=node_sizes[:l], node_color=viridis_palette[:l])
    nx.draw_networkx_edges(G1, pos, edgelist=elarge, width=width, edge_color="r")
    nx.draw_networkx_edges(
        G1, pos, edgelist=esmall, width=width, alpha=0.5, edge_color="b"
    )

    # node labels
    nx.draw_networkx_labels(G1, pos, font_family="sans-serif", font_size=20, font_color='white', font_weight='bold')
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G1, "weight")
    nx.draw_networkx_edge_labels(G1, pos, edge_labels, label_pos=0.6)
    # ax.text(0.1, 1.05, 'b', transform=ax.transAxes,
    #   fontsize=16, fontweight='bold', va='top', ha='right')

    ax = plt.gca()
    ax.margins(0.08)

    plt.axis("off")
    # plt.title(f'Cell - {cell}_after')
    plt.tight_layout()
    
    plt.savefig(f'{path}/{cell}_network_changes_after.svg')
    # plt.show()
    plt.close()


    plt.figure(figsize=(15,15))

    ax = plt.subplot(1,1,1)
    elarge = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] > 0]
    esmall = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] <= 0]

    # pos = nx.spring_layout(G2) 
    
    # nx.draw_networkx_nodes(G2, pos, node_size=700)
    l = G2.number_of_nodes()
    nx.draw_networkx_nodes(G2, pos, node_size=node_sizes[:l], node_color=viridis_palette[:l])
    nx.draw_networkx_edges(G2, pos, edgelist=elarge, width=width, edge_color="r")
    # nx.draw_networkx_edges(
    #     G2, pos, edgelist=esmall, width=width
    # )
    nx.draw_networkx_edges(
        G2, pos, edgelist=esmall, width=width, alpha=0.5, edge_color="b"
    )

    # node labels
    nx.draw_networkx_labels(G2, pos, font_family="sans-serif", font_size=20, font_color='white', font_weight='bold')
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G2, "weight")
    # nx.draw_networkx_edges(G, pos, ax=ax, edge_labels=edge_labels)
    nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels, label_pos=0.6)

    ax = plt.gca()
    ax.margins(0.08)

    plt.axis("off")
    # plt.title(f'Cell - {cell}_before')
    plt.tight_layout()
    
    
    plt.savefig(f'{path}/{cell}_network_before.svg')
    # plt.show()
    plt.close()


    print(G1.number_of_edges(), G2.number_of_edges())

    return G1, G2


def regularization_term(matrix, reference_matrix, lambda_):
    diff_norm = torch.norm(matrix - reference_matrix, p=2)
    return lambda_ * diff_norm


def count_same_sign_excluding_zero(arr1, arr2):
    # Compute the signs of elements in arr1 and arr2
    sign_arr1 = np.sign(arr1)
    sign_arr2 = np.sign(arr2)
    
    # Exclude positions where both arr1 and arr2 are 0
    non_zero_mask = (arr1 != 0) | (arr2 != 0)
    sign_arr1_filtered = sign_arr1[non_zero_mask]
    sign_arr2_filtered = sign_arr2[non_zero_mask]
    
    # Compare signs and count positions where signs are the same
    same_sign_count = np.sum(sign_arr1_filtered == sign_arr2_filtered)
    
    return same_sign_count

def count_different_sign_positions(predicted, actual):
    # Create boolean masks for different sign conditions
    mask1 = (predicted > 0) & (actual <= 0)  # predicted > 0 and actual <= 0
    mask2 = (predicted < 0) & (actual >= 0)  # predicted < 0 and actual >= 0
    
    # Combine masks using logical OR (|)
    different_sign_mask = mask1 | mask2  # Positions where signs are different
    
    # Count the number of True values in the different_sign_mask
    num_different_sign_positions = np.sum(different_sign_mask)
    
    return num_different_sign_positions

def count_both_zero_positions(predicted, actual):
    # Create boolean mask for positions where both arrays are 0
    both_zero_mask = (predicted == 0) & (actual == 0)
    
    # Count the number of True values in the both_zero_mask
    num_both_zero_positions = np.sum(both_zero_mask)
    
    return num_both_zero_positions

def count_predicted_zero_actual_nonzero(predicted, actual):
    # Create boolean mask for positions where predicted is 0 and actual is not 0
    predicted_zero_actual_nonzero_mask = (predicted == 0) & (actual != 0)
    
    # Count the number of True values in the mask
    num_positions = np.sum(predicted_zero_actual_nonzero_mask)
    
    return num_positions


def count_predicted_actual_nonzero(predicted, actual):
    # Create boolean mask for positions where predicted is 0 and actual is not 0
    predicted_actual_nonzero_mask = (predicted != 0) & (actual != 0)
    
    # Count the number of True values in the mask
    num_positions = np.sum(predicted_actual_nonzero_mask)
    
    return num_positions


def count_predicted_nonzero_actual_zero(predicted, actual):
    # Create boolean mask for positions where predicted is 0 and actual is not 0
    predicted_nonzero_actual_zero_mask = (predicted != 0) & (actual == 0)
    
    # Count the number of True values in the mask
    num_positions = np.sum(predicted_nonzero_actual_zero_mask)
    
    return num_positions


def calculate_metrics(predicted, actual):
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = count_same_sign_excluding_zero(predicted, actual)
    FP = count_different_sign_positions(predicted, actual)
    TN = count_both_zero_positions(predicted, actual)
    FN = count_predicted_zero_actual_nonzero(predicted, actual)
    
    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate F1 Score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


def calculate_metrics_positive(predicted, actual):
    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
    TP = count_predicted_actual_nonzero(predicted, actual)
    FP = count_predicted_nonzero_actual_zero(predicted, actual)
    TN = count_both_zero_positions(predicted, actual)
    FN = count_predicted_zero_actual_nonzero(predicted, actual)

    print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
    
    # Calculate Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Calculate Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # Calculate F1 Score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def calculate_metrics_binary(predicted, actual):
    flat_matrix_1 = predicted.flatten()
    flat_matrix_2 = actual.flatten()

    accuracy = accuracy_score(flat_matrix_2, flat_matrix_1)
    f1 = f1_score(flat_matrix_2, flat_matrix_1)
    mcc = matthews_corrcoef(flat_matrix_2, flat_matrix_1)

    return f1, accuracy, mcc




def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory {dir_path} created.")
    else:
        print(f"Directory {dir_path} already exists.")