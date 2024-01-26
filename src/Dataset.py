from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):

  def __init__(self,data,word2idx,device=torch.device('cpu')):
    # data=pd.read_csv(file_name,index_col=[0])

    x=data.iloc[:,0:-1].values
    y=data.iloc[:,-1].values

    self.x_train=torch.tensor(x,dtype=torch.float32,device=device)
    self.y_train= torch.tensor([word2idx[p] for p in y], dtype=torch.long,device=device)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]
  
