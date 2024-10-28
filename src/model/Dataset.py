from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


class MyDataset(Dataset):

  def __init__(self,data,word2idx,x_type=torch.float32, y_type=torch.long,device=torch.device('cpu')):

    x=data.iloc[:,0:-1].values
    y=data.iloc[:,-1].values

    self.x_train=torch.tensor(x,dtype=x_type,device=device)
    labels = torch.tensor([word2idx[p] for p in y], dtype=y_type,device=device)

    # Determine the number of classes (assuming labels are 0-indexed)
    num_classes = labels.max().item() + 1

    # print(num_classes, len(labels))

    # Create one-hot encoded vectors
    one_hot_labels = torch.zeros(len(labels), num_classes, device=device, dtype=y_type)
    # print(labels)
    # print(labels.dim())

    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

    self.y_train = one_hot_labels

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]
  

class ContrastiveDataset(Dataset):
    def __init__(self, dataset, device=torch.device('cpu')):
        self.dataset = dataset.to(device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx]
        x_i = self.augment(x)  # Augmentation 1
        x_j = self.augment(x)  # Augmentation 2
        return x_i, x_j

    def augment(self, x):
        # Apply random augmentations like cropping, noise, etc.
        if x.dtype != torch.float:
           x = x.float()
        return x + torch.randn_like(x) * 0.1  # Add random noise as an example
    

# Dataset class to handle gene expression data
class GeneExpressionDataset(Dataset):
    def __init__(self, expression_data, random_inputs, labels=None):
        self.expression_data = expression_data
        self.random_inputs = random_inputs
        self.labels = labels

    def __len__(self):
        return len(self.expression_data)

    def __getitem__(self, idx):
        if self.labels.any():
           return self.expression_data[idx], self.random_inputs[idx], self.labels[idx]
        else:
            return self.expression_data[idx], self.random_inputs[idx]