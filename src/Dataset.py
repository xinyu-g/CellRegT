from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):

  def __init__(self,data,word2idx,x_type=torch.float32, y_type=torch.long,device=torch.device('cpu')):

    x=data.iloc[:,0:-1].values
    y=data.iloc[:,-1].values

    self.x_train=torch.tensor(x,dtype=x_type,device=device)
    self.y_train= torch.tensor([word2idx[p] for p in y], dtype=y_type,device=device)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]
  
