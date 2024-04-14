import torch
import json
import argparse
import torch.nn as nn
from os import path
import pandas as pd
import numpy as np
from utils import split_df, onehot
from Dataset import MyDataset
from torch.utils.data import DataLoader
from LinearRegression import LinearRegressionModel, train_linear_model




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Database Path')
    parser.add_argument('-i', help='intput scRNAseq file path', type=str, default="../data/example_data.csv")
    parser.add_argument('-r', help='intput RN file path', type=str, default="../data/example_rn.csv")
    parser.add_argument('-o', help='output file path', type=str, default="../data/ranked_genes.json")
    parser.add_argument('-O', help='output file path', type=str, default="../data/expression_w_ranked_genes.csv")
    parser.add_argument('-e', help='# epochs', type=int, default=500)
    parser.add_argument('-b', help='batch size', type=int, default=200)
    parser.add_argument('-l', help='learning rate', type=float, default=0.01)
    parser.add_argument('-d', help='dataset name', type=str, default='')


    args = parser.parse_args()
    input = path.realpath(args.i)
    # rn = path.realpath(args.r)
    output = path.realpath(args.o)
    output_csv = path.realpath(args.O)
    n_epochs = args.e
    batch_size = args.b
    lr = args.l
    dname = args.d


    seq_data = pd.read_csv(input, index_col=[0], compression='gzip')
    # regulatory_data = pd.read_csv(rn, index_col=[0])

    # genes = np.intersect1d(seq_data.columns, regulatory_data.columns)
    # genes = genes.tolist()
    genes = seq_data.columns.to_list()
    data = seq_data.copy()

    # number of clusters
    # k = 10

    # the genes cols + the phenotype col
    # cols = genes + [f'K = {str(k)}']



    # data = data[cols]
    cell_map_data = pd.read_csv(f'/Users/xinyu/Documents/Wang/data/{dname}/cell_mapping.csv', index_col=[0])
    # cell_map_data = pd.read_csv('/Users/xinyu/Documents/Wang/data/smartSeq_Mouse/cell_mapping.csv', index_col=[0])

    cell_map = dict(zip(cell_map_data.index, cell_map_data['cluster_label']))

    data['clusters'] = data.index.map(cell_map)

    clusters = data['clusters'].unique()

    k = len(clusters)

    idx = 0
    word2idx = {}
    for p in clusters:
        word2idx[p] = onehot(idx, k)
        idx += 1

    cluster_map = open(f'../models/data/{dname}_cluster_map.json', 'w')
    json.dump(word2idx, cluster_map)
    cluster_map.close()

    # mps_device = torch.device("mps")

    train, valid = split_df(0.9, data)
    train_dataset = MyDataset(train, word2idx)
    valid_dataset = MyDataset(valid, word2idx)


    Z = torch.randint(2, (batch_size, k * len(genes)), dtype=torch.float32)
    Z[:, 0] = 1

    # mps_device = torch.device("mps")

    model = LinearRegressionModel(len(genes), len(genes)*k, k)
    # model.to(mps_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    BATCH_SIZE = batch_size
    train_loader = DataLoader(train_dataset, BATCH_SIZE,drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, BATCH_SIZE,drop_last=True, shuffle=False)

    train_loss, valid_loss = train_linear_model(train_loader, valid_dataset, model, lr=lr)

    coefficients = model.state_dict()['linear.weight'].numpy()
    features = genes
    coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': np.abs(coefficients[0])})
    coefficients_df['Coefficient_Abs'] = np.abs(coefficients_df['Coefficient'])
    coefficients_df.sort_values(by='Coefficient_Abs', ascending=False, inplace=True)
    features = coefficients_df['Feature'].to_list()
    genes_file = open(output, 'w')
    json.dump(features, genes_file)
    genes_file.close()

    data[features[:1000] + ['clusters']].to_csv(output_csv, index=True)



