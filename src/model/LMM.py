import torch
import json
import argparse
import torch.nn as nn
from os import path
import pandas as pd
import numpy as np
from .utils import *
from .Dataset import MyDataset
from torch.utils.data import DataLoader



class SupervisedLinearMixedModel(nn.Module):
    def __init__(self, input_dim, latent_dim, random_effect_dim, output_dim):
        super(SupervisedLinearMixedModel, self).__init__()
        # Fixed effect: Linear layer to learn global representation
        self.fixed_effect = nn.Linear(input_dim, latent_dim)

        # Random effect: Learn variability per sample or group
        self.random_effect = nn.Linear(random_effect_dim, latent_dim)

        # Output layer for supervised learning
        self.output_layer = nn.Linear(latent_dim, output_dim)

    def forward(self, X, random_inputs):
        # Learn fixed effect representation
        fixed_rep = self.fixed_effect(X)
        
        # Learn random effect representation
        random_rep = self.random_effect(random_inputs)
        
        # Combine fixed and random effects
        combined_rep = fixed_rep + random_rep
        
        # Output layer for prediction
        output = self.output_layer(combined_rep)
        return output
    


def train_supervised_model(train_data, model, num_epochs=500, lr=0.01, criterion=nn.CrossEntropyLoss()):

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0.0

        for i, (src, random_inputs, labels) in enumerate(train_data):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(src, random_inputs)
            
            # Compute supervised loss
            loss = criterion(predictions, labels)
            train_total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
        if epoch % 50 == 1:
            print(f"Epoch {epoch}, Train Loss: {train_total_loss / len(train_data)}")

    print("Supervised training completed.")
    return model


class SelfSupervisedLinearMixedModel(nn.Module):
    def __init__(self, input_dim, latent_dim, random_effect_dim):
        super(SelfSupervisedLinearMixedModel, self).__init__()
        # Fixed effect: Linear layer to learn global representation
        self.fixed_effect = nn.Linear(input_dim, latent_dim)

        # Random effect: Learn variability per sample or group
        self.random_effect = nn.Linear(random_effect_dim, latent_dim)

    def forward(self, X, random_inputs):
        # Learn fixed effect representation
        fixed_rep = self.fixed_effect(X)
        
        # Learn random effect representation
        random_rep = self.random_effect(random_inputs)
        
        # Combine fixed and random effects
        combined_rep = fixed_rep + random_rep
        return combined_rep  # This will be the learned representation
    

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        # Normalize the representations
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Compute similarity scores
        logits = torch.matmul(z_i, z_j.T) / self.temperature

        # Create labels, where the i-th sample matches the i-th sample in the second batch
        labels = torch.arange(batch_size).to(logits.device)

        # Cross-entropy loss on logits
        loss = self.criterion(logits, labels)
        return loss
    

def augment_data(expression_data, noise_factor=0.1):
    if expression_data.dtype != torch.float:
        expression_data = expression_data.float()
    # Add random Gaussian noise to the input data
    noise = torch.randn_like(expression_data) * noise_factor
    return expression_data + noise


def augment_data_mask(expression_data, mask_fraction=0.1):
    if expression_data.dtype == torch.long:
        expression_data = expression_data.float()
    # Mask a random subset of genes by setting their values to zero
    mask = torch.rand_like(expression_data) > mask_fraction
    return expression_data * mask


def train_self_supervised_model(train_data, model, num_epochs=500, lr=0.01):

    # Contrastive loss function
    criterion = ContrastiveLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # print('Epoch:', epoch)
        model.train()
        train_total_loss = 0.0

        for i, (src, random_inputs) in enumerate(train_data):
            # Create two augmented views of the input (for contrastive learning)
            augmented_view1 = augment_data(src)
            augmented_view2 = augment_data(src)


            # Forward pass for both views
            optimizer.zero_grad()
            rep_view1 = model(augmented_view1, random_inputs)
            rep_view2 = model(augmented_view2, random_inputs)

            # Compute contrastive loss
            loss = criterion(rep_view1, rep_view2)
            train_total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        if epoch % 50 == 1:
            print(f"Epoch {epoch}, Train Loss: {train_total_loss / len(train_data)}")

    print("Training completed.")
    return model

def predict(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for src, random_inputs in data_loader:
            output = model(src, random_inputs)
            predictions.append(output)

    return torch.cat(predictions, dim=0)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src, random_inputs, labels in data_loader:
            predictions = model(src, random_inputs)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation Loss: {avg_loss}")
    return avg_loss


def save_model(model, filepath="model.pth"):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath="model.pth"):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")


def prepare_dataloader(data, batch_size=32, shuffle=True):
    dataset = MyDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def extract_fixed_importance(model, aggregation_method='mean'):
    # The fixed effect layer coefficients (two-dimensional: [latent_dim, num_features])
    fixed_coefficients = model.fixed_effect.weight.detach().cpu().abs()

    # Aggregate across latent dimensions (e.g., using mean, sum, or max)
    if aggregation_method == 'mean':
        feature_importance = fixed_coefficients.mean(dim=0)  # Take mean across latent dimensions
    elif aggregation_method == 'sum':
        feature_importance = fixed_coefficients.sum(dim=0)   # Take sum across latent dimensions
    elif aggregation_method == 'max':
        feature_importance = fixed_coefficients.max(dim=0)[0]  # Take max across latent dimensions
    else:
        raise ValueError("Invalid aggregation method. Choose from 'mean', 'sum', or 'max'.")

    # Rank the features by importance (in descending order)
    ranked_indices = torch.argsort(feature_importance, descending=True)

    return feature_importance, ranked_indices