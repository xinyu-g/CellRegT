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
from Transformer import NoamOpt, shape_rn, onehot, make_classification_model, run_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Database Path')
    parser.add_argument('-i', help='intput scRNAseq file path', type=str, default="../data/example_data.csv")
    parser.add_argument('-r', help='intput RN file path', type=str, default="../data/example_rn.csv")
    parser.add_argument('-g', help='intput gene rank file path', type=str, default="../data/ranked_genes.json")
    parser.add_argument('-o', help='output file path', type=str, default="../data/example_output.pt")
    parser.add_argument('-e', help='# epochs', type=int, default=200)
    parser.add_argument('-b', help='batch size', type=int, default=100)
    parser.add_argument('-l', help='learning rate', type=float, default=0.01)
    parser.add_argument('-k', help='top k genes', type=float, default=100)
    parser.add_argument('-N', help='# of layers with RN', type=int, default=1)
    parser.add_argument('-c', help='colname of clusters', type=str, default='clusters')
    parser.add_argument('-d', help='device', type=str, default='cpu')


    args = parser.parse_args()
    input = path.realpath(args.i)
    gene_rank = path.realpath(args.g)
    rn = path.realpath(args.r)
    output = path.realpath(args.o)
    n_epochs = args.e
    batch_size = args.b
    lr = args.l
    n = args.k
    N = args.N
    colname = args.c
    device = torch.device(args.d)

    data = pd.read_csv(input, index_col=[0])
    regulatory_data = pd.read_csv(rn, index_col=[0])


    genes = np.intersect1d(data.columns, regulatory_data.columns)
    genes = genes.tolist()

    f = open(gene_rank, 'r')
    ranked_genes = json.load(f)
    f.close()

    remove = []
    for gene in ranked_genes:
        if gene not in genes:
            remove.append(gene)

    for gene in remove:
        ranked_genes.remove(gene)

    rn_more_genes = ranked_genes.copy()
    rn_genes = regulatory_data.index.to_list()
    for gene in rn_genes:
        if gene in rn_more_genes:
            rn_more_genes.remove(gene)
    rn_more = pd.DataFrame(0, index=rn_more_genes, columns=ranked_genes)
    rn = regulatory_data[ranked_genes]
    rn = pd.concat([rn, rn_more])

    genes = ranked_genes[:n]
    RN = rn.loc[genes]
    RN = RN[genes]
    dh = 10

    RN = torch.from_numpy(RN.to_numpy().astype(np.float32))
    RN = shape_rn(RN, n//dh, dh, batch_size)
    RN.requires_grad=True

    # number of clusters
    k = 10

    seq_data = data[genes + [colname]]
    phenotypes = data[colname].unique()

    vocab = []
    for gene in genes:
        vocab.extend(pd.unique(seq_data[gene]))
    vocab = list(set(vocab))

    idx = 0
    word2idx = {}
    for p in phenotypes:
        word2idx[p] = idx
        idx += 1

    train, valid = split_df(0.9, seq_data)
    train_dataset = MyDataset(train, word2idx)
    valid_dataset = MyDataset(valid, word2idx)

    
    BATCH_SIZE = batch_size

    model = make_classification_model(len(vocab), len(phenotypes), RN, d_model=n, h=dh, N2=N)
    model.to(device)

    is_rn = True 

    if not N:
        is_rn = False
    y_pred, y_true = run_model(model,RN,n_epochs,BATCH_SIZE,train_dataset,valid_dataset,lr, is_rn)

    torch.save(model, output)