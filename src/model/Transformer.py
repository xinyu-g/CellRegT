import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from .utils import *


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def clones(module, N, rn_module=None, N2=None):
    "Produce N identical layers."
    if N2:
        return nn.ModuleList([copy.deepcopy(rn_module) for _ in range(N2)] + [copy.deepcopy(module) for _ in range(N)])
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, RN_layer=None, N2=None):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N, RN_layer, N2)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
    

class EncoderClass(nn.Module):
    """
    
    """
    def __init__(self, encoder, src_embed=None, tgt_embed=None, generator=None):
        super(EncoderClass, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        rn_weights = self.encoder.layers[0].RN if self.encoder.layers[0].rn else None
        output = self.encode(src, src_mask)
        return output, rn_weights
    
    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
    

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout, RN=None, rn=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.rn = rn
        self.RN = None
        if self.rn:
            # self.RN = nn.Parameter(torch.sigmoid(RN))
            # self.RN = nn.Parameter(RN)
            self.RN = RN

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        #
        RN = self.RN() if self.rn else None
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, RN, mask))
        return self.sublayer[1](x, self.feed_forward)
    

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    

def attention(query, key, value, RN=None, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # print('attn')
    d_k = query.size(-1)
    # print(query.shape, key.transpose(-2, -1).shape)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print(scores.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = 1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print(1, scores.shape, p_attn.shape)
    return torch.matmul(p_attn, value), p_attn


def attention_w_regulatoryNet(query, key, value, rn=None, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(torch.matmul(rn, query.unsqueeze(-1)).squeeze(-1), key.transpose(-2,-1))\
             / math.sqrt(d_k)
    
    # print("scores:", scores.shape)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    
    p_attn = F.softmax(scores, dim=1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def shape_rn(t, d_k, h, batch):
    diagonal_tensors = []

# Iterate to get the d_k diagonal tensors
    for i in range(h):
        # Calculate the starting and ending indices for the diagonal tensor
        start_idx = i * d_k
        end_idx = start_idx + d_k
        
        # Extract the diagonal tensor using torch.diagonal
        diagonal_tensor = t[start_idx:end_idx, start_idx:end_idx]
        
        # Append the diagonal tensor to the list
        diagonal_tensors.append(diagonal_tensor)
    r1 = torch.stack(diagonal_tensors, dim=0)


    r2 = r1.repeat(batch, 1, 1)
    if h == 1:
        return r2
    # print(r2.shape)
    r2 = r2.view(batch, h, 1, d_k, d_k)
    return r2


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, RN=None, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)


        if self.h == 1:
            query, key, value = \
                [l(x).view(nbatches, -1)
                 for l, x in zip(self.linears, (query, key, value))]
        else:
            query, key, value = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))] 
        
        # 2) Apply attention on all the projected vectors in batch. 
        if torch.is_tensor(RN):
        # if RN:

            x, self.attn = attention_w_regulatoryNet(query, key, value, RN, mask=mask, 
                                 dropout=self.dropout)
            
        else:
            x, self.attn = attention(query, key, value, mask=mask, 
                                    dropout=self.dropout)
            
        if self.h != 1:
            x = x.transpose(1, 2).contiguous() \
                .view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x).squeeze(1)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class RNModule(nn.Module):
    def __init__(self, initial_value, tanh=None):
        super(RNModule, self).__init__()
        # Register the parameter using nn.Parameter
        self.RN = nn.Parameter(initial_value) if torch.is_tensor(initial_value) else None
        self.tanh = torch.tanh if tanh else None

    def forward(self):
        # The forward pass can be as simple as returning the parameter or using it in some operation
        return self.tanh(self.RN) if torch.is_tensor(self.RN) and self.tanh else self.RN
        # return torch.tanh(self.RN) if torch.is_tensor(self.RN) else None
    

def make_classification_model(tgt_vocab, RN=None, N=6, 
               d_model=512, d_ff=2048, h=1, dropout=0.1, N2=0, p_name='encoder.encoder.layers.0.RN.RN', lambda_=1, tanh=''):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    attn_w_RN = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = TransformerClassifier(
        EncoderClass(
            Encoder(
                EncoderLayer(d_model, c(attn), c(ff), dropout),
                N-N2,
                EncoderLayer(d_model, c(attn_w_RN), c(ff), dropout, RNModule(RN, tanh), True),
                N2
            ),
            tgt_embed=None
        ),
        d_model,
        tgt_vocab,
        lambda_=lambda_
    )
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for name, param in model.named_parameters():
        # print(name)
        if param.dim() > 1 and name != p_name:  # Check if parameter is a weight matrix 
            nn.init.xavier_uniform_(param)

    if N2:
        print(RN == model.state_dict()[p_name])
    return model


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1):
        """
        Constructor for the label smoothing loss.
        :param smoothing: smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        Forward pass for label smoothing.
        :param pred: Predictions from the model (logits, not softmax)
        :param target: Ground truth labels, class indices
        :return: Smoothed loss
        """
        # Get the number of classes
        num_classes = pred.size(-1)
        
        # Convert target to one-hot encoding
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        
        # Apply log softmax to predictions and calculate the smoothed loss
        log_prob = F.log_softmax(pred, dim=-1)
        loss = torch.mean(torch.sum(-true_dist * log_prob, dim=-1))
        
        return loss

    
class LabelSmoothingCrossEntropy(nn.Module):
    """
    CrossEntropyLoss with label smoothing.
    """
    def __init__(self, num_classes, padding_idx, smoothing=0.0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.padding_idx = padding_idx
        self.num_classes = num_classes

    def forward(self, x, target):
        # Apply log_softmax to the input logits
        log_probs = F.log_softmax(x, dim=-1)

        # Create the smoothed label distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            # Handle padding index, if any
            if self.padding_idx is not None:
                true_dist[:, self.padding_idx] = 0
                mask = target == self.padding_idx
                true_dist[mask] = 0
        
        # Compute the cross-entropy loss with the smoothed labels
        loss = torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

        return loss
    

class TransformerClassifier(nn.Module):
    def __init__(self, encoder, d_model, targets, lambda_=0.5):
        super(TransformerClassifier, self).__init__()
        self.encoder = encoder  # Your transformer encoder
        self.dropout = nn.Dropout(p=0.3)  # Optional dropout
        self.output_layer = nn.Linear(d_model, targets)  # Linear layer for class projection
        self.previous_rn_weights = None
        self.lambda_ = lambda_
    
    def forward(self, x, mask=None):
        x, rn_weights = self.encoder(x, mask)  # x will have shape [200, 500]
        x = self.dropout(x)
        logits = self.output_layer(x)  # Projected to [200, 40]
        return logits, rn_weights
    
    def regularization_loss(self, rn_weights):
        if self.previous_rn_weights is None:
            self.previous_rn_weights = rn_weights.detach().clone()
        # print(1, rn_weights == self.previous_rn_weights)
        return regularization_term(rn_weights, self.previous_rn_weights, self.lambda_)
    
    def update_reference(self, rn_weights):
        self.previous_rn_weights = rn_weights.detach().clone()


def run_epoch(data_iter, model, criterion, opt=None):
    "Standard Training and Logging Function"
    total_loss = 0
    y_pred = []
    y_true = []
    y_pred_prob = []
    y_true_prob = []
    for i, (src, tgt) in enumerate(data_iter):

        out, rn_weights =  model.forward(src, None) #

        target = torch.argmax(tgt.float(), dim=1)
        reg_loss = model.regularization_loss(rn_weights) if torch.is_tensor(rn_weights) else 0
        # print(f'reg: {model.encoder.lambda_} reg loss: {reg_loss}')

        loss = criterion(out, target) + reg_loss
        loss.backward()
        if opt is not None:
            opt.step()
            opt.optimizer.zero_grad()
        
        y_p = torch.argmax(out, dim=1)
        y_pred.append(y_p)
        y_pred_prob.append(out)
        trg_y = tgt
        y_true.append(trg_y)
        y_true_prob.append(tgt)
        total_loss += loss

    return total_loss, y_pred_prob, y_true_prob


def run_model(model,RN,n_epoch,batch_size,train_dataset,valid_dataset,d_model, lr=0.000001, rn=True, path=''):
    criterion = LabelSmoothingLoss()
    # criterion = nn.KLDivLoss(reduction='batchmean')
    model_opt = NoamOpt(d_model, 1, 2000,
                torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))
    BATCH_SIZE = batch_size
    train_loader = DataLoader(train_dataset, BATCH_SIZE, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, BATCH_SIZE, drop_last=True, shuffle=False)
    best_train_loss, best_valid_loss = float('inf'), float('inf')
    epoch, cnt = 0, 0
    train_loss = 0
    print(f'reg: {model.lambda_}')
    while epoch < n_epoch and cnt < 10:

        if train_loss <= best_train_loss:
            model.train()
            train_loss, _, _ = run_epoch( train_loader,
                        model, 
                        criterion,
                        model_opt)
            
            best_train_loss = train_loss
            
            epoch += 1
            cnt = 0
        else:
            cnt += 1

        if epoch % 20 == 1:
            print(f"Epoch {epoch}, cnt: {cnt}")

        # eval
        model.eval()

        valid_loss, y_pred_eval, y_true_eval  = run_epoch( valid_loader, 
                            model, 
                            criterion,
                            model_opt)

        if valid_loss <= best_valid_loss:
            y_pred_return, y_true_return = y_pred_eval, y_true_eval
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), path)

        
        if epoch % 20 == 1:
            print(f"Epoch {epoch}, Train average Loss: {train_loss}, Valid average Loss: {valid_loss}" )
        
    return y_pred_return, y_true_return
