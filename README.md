# CellRegT

Dynamic Inference of Cell-Type-Specific GRNs with Integrated Regulatory Priors


## Installation

### Build from source

```
git clone https://github.com/xinyu-g/CellRegT.git
cd CellRegT
conda env create -f CellRegT.yml
conda activate CellRegT

cd src
```

### Run LMM to select top K genes

change LMMPaths parameters in /src/config.yaml

```
LMMPaths:

  input_scRNAseq_train: <path to input scRNAseq data (train set)>
  input_scRNAseq_test: <path to input scRNAseq data (test set)>
  output_rn_file: <output path to save prior RN>
  output_ranked_genes_file: <output path to save ranked genes>
  output_expression_train: <output path to save scRNAseq data with top n ranked genes (train set)>
  output_expression_test: <output path to save scRNAseq data with top n ranked genes (test set)>
  cluster_map_output: <output path to save cluster look up table>
  model_output: <output path to save the LMM model>
  db_path: <path to the CollecTRI database> ("../data/CollecTRI.csv")
```

Other parameters you might want to change


```
dataset:
  name: <Name of dataset> (optional)
  label_col: <Name of the column that stores labels>
  device: <device: cpu, gpu, etc> 
```

```
training: (all training read from this, you might want to change the parameters when training different models)
  epochs: <number of epochs>
  batch_size: <batch size>
  model_lr: <learning rate>
```

```
LMMmodel:
  supervised: <1 for supervised model, 0 for self-supervised model>
  latent_dim: <latent dimension>
  num_random_effects: <number of random effects>
```

```
data_processing:
  top_features_to_save: <Top N features to save in output CSV>
  top_k: <top k genes to construct prior RN>
  get_RN: <get prior from the STRING database or CollecTri database if 1>
```

#### Command

```
python ./src/train_LMM.py
```



### Run Transformer Model

if you skipped the LMM selection step, you need to make sure some of the paths in LMMPaths are properly filled

```
LMMPaths:
  output_rn_file: <Pre-defined rn provided by user>
  output_ranked_genes_file: <List of genes of user interest> 
  output_expression_train: <Path to input scRNAseq data (train set)>
  output_expression_test: <Path to input scRNAseq data (test set)>
  cluster_map_output: <A cluster look up table: map cell type to idx>
```

```
dataset:
  label_col: <Name of the column that stores labels>
```

Other parameters you might want to change 

```
TransformerModel:
  supervised: <1 for supervised mode, 0 for self-supervised model>
  latent_dim: <latent dimension>
  num_layers: <number of layers to integrate RN priors>
```

```
data_processing:
  top_k: <number of genes to include/number of genes in prior RN>
```

#### Command

```
python ./src/train.py
```


## Run Inference

#### Command

```
python ./src/inference.py
```
