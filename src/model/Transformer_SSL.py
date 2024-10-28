# Some layers are adapted from Harvard NLP's "Annotated Transformer" repository
# Original source: https://github.com/harvardnlp/annotated-transformer
# License: MIT License


import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from .utils import *
from .Dataset import *
from .Transformer import *

class ProjectionHead(nn.Module):
    def __init__(self, d_model, projection_dim):
        super(ProjectionHead, self).__init__()
        self.fc = nn.Linear(d_model, projection_dim)

    def forward(self, x):
        return nn.functional.normalize(self.fc(x), dim=1)  # Normalize for contrastive learning
    

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
    

class EncoderClass(nn.Module):
    def __init__(self, encoder, src_embed=None, tgt_embed=None, generator=None, projection_dim=128):
        super(EncoderClass, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        # Add projection head for contrastive learning
        self.projection_head = ProjectionHead(d_model=encoder.layers[0].size, projection_dim=projection_dim)

    def forward(self, src, tgt):
        # Encoder pass
        encoded = self.encoder(src)
        # Apply projection head for contrastive learning
        projected = self.projection_head(encoded)
        return projected
    


def make_self_supervised_model(RN=None, N=6, 
                               d_model=512, d_ff=2048, h=1, dropout=0.1, N2=0, 
                               p_name='encoder.layers.0.RN.RN', lambda_=1, tanh='', projection_dim=128):
    "Helper: Construct a self-supervised model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    attn_w_RN = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # Use EncoderClass with Projection Head for contrastive learning
    model = EncoderClass(
        encoder=Encoder(
            EncoderLayer(d_model, c(attn), c(ff), dropout),
            N-N2,
            EncoderLayer(d_model, c(attn_w_RN), c(ff), dropout, RNModule(RN, tanh), True),
            N2
        ),
        # src_embed=nn.Sequential(Embeddings(d_model, d_model)),  # Embedding layer (can be unchanged)
        projection_dim=projection_dim  # Add projection head
    )

    # Initialize parameters with Glorot / fan_avg.
    for name, param in model.named_parameters():
        if param.dim() > 1 and name != p_name:  # Check if parameter is a weight matrix 
            nn.init.xavier_uniform_(param)

    if N2:
        print(RN == model.state_dict()[p_name])
    
    return model
    

def run_epoch(data_loader, model, contrastive_loss, optimizer=None):
    total_loss = 0

    for batch in data_loader:
        x_i, x_j = batch  # Get two augmented views

        if optimizer:
            optimizer.zero_grad()

        # Forward pass through the encoder and projection head
        z_i = model(x_i, x_i)  # Pass through encoder and projection head
        z_j = model(x_j, x_j)

        # Compute contrastive loss
        loss = contrastive_loss(z_i, z_j)

        if optimizer:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def train_contrastive_model(model, train_dataset, n_epochs=10, batch_size=64, lr=1e-3, save_path='best_model.pth'):
    # Create contrastive dataset and dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize loss function and optimizer
    contrastive_loss = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Variable to store the best loss and model
    best_loss = float('inf')  # Initialize with infinity
    save_count = 0 

    # Training loop
    for epoch in range(n_epochs):
        train_loss = run_epoch(train_loader, model, contrastive_loss, optimizer)

        if epoch % 20 == 1:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss:.4f}")
        
        # Check if the current epoch's loss is better than the best loss
        if train_loss < best_loss:
            best_loss = train_loss
            # Save the model with the best loss
            torch.save(model.state_dict(), save_path)
            
            save_count += 1
            
            # Print the message only every 20 saves
            if save_count % 20 == 0:
                print(f"Model saved with loss {best_loss:.4f} at epoch {epoch + 1}")