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
  
