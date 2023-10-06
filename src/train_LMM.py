import torch
import json
import argparse
import torch.nn as nn
from os import path
import pandas as pd
import numpy as np
from utils import split_df
from Dataset import MyDataset
from torch.utils.data import DataLoader
from LMM import LinearMixedModel, train_linear_mixed_model




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Database Path')
    parser.add_argument('-i', help='intput scRNAseq file path', type=str, default="../data/example_data.csv")
    parser.add_argument('-r', help='intput RN file path', type=str, default="../data/example_rn.csv")
    parser.add_argument('-o', help='output file path', type=str, default="../data/ranked_genes.json")
    parser.add_argument('-e', help='# epochs', type=int, default=500)
    parser.add_argument('-b', help='batch size', type=int, default=200)
    parser.add_argument('-l', help='learning rate', type=float, default=0.01)


    args = parser.parse_args()
    input = path.realpath(args.i)
    rn = path.realpath(args.r)
    output = path.realpath(args.o)
    n_epochs = args.e
    batch_size = args.b
    lr = args.l


    seq_data = pd.read_csv(input, index_col=[0])
    regulatory_data = pd.read_csv(rn, index_col=[0])

    genes = np.intersect1d(seq_data.columns, regulatory_data.columns)
    genes = genes.tolist()
    data = seq_data.copy()

    # number of clusters
    k = 10

    # the genes cols + the phenotype col
    cols = genes + [f'K = {str(k)}']
    data = data[cols]

    train, valid = split_df(0.9, data)
    train_dataset = MyDataset(train)
    valid_dataset = MyDataset(valid)


    Z = torch.randint(2, (batch_size, k * len(genes)), dtype=torch.float32)
    Z[:, 0] = 1

    model = LinearMixedModel(len(genes), len(genes)*k)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    BATCH_SIZE = batch_size
    train_loader = DataLoader(train_dataset, BATCH_SIZE,drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, BATCH_SIZE,drop_last=True, shuffle=False)

    train_loss, valid_loss = train_linear_mixed_model(train_loader, valid_dataset, model, Z, lr=lr)

    coefficients = model.state_dict()['fixed_effects.weight'].numpy()
    features = genes
    coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': np.abs(coefficients[0])})
    coefficients_df['Coefficient_Abs'] = np.abs(coefficients_df['Coefficient'])
    coefficients_df.sort_values(by='Coefficient_Abs', ascending=False, inplace=True)
    features = coefficients_df['Feature'].to_list()
    genes_file = open(output, 'w')
    json.dump(features, genes_file)
    genes_file.close()


