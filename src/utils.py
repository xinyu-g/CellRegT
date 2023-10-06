from torch.utils.data import random_split
  

def split_df(ratio, data):
    train_dataset = data.sample(frac=ratio)
    valid_dataset = data.drop(train_dataset.index)
    return train_dataset, valid_dataset

def split(ratio, data):
    train_set_size = int(len(data) * ratio)
    valid_set_size = len(data) - train_set_size
    train_set, valid_set = random_split(data, [train_set_size, valid_set_size])
    return train_set, valid_set

def onehot(idx, length):
    lst = [0 for i in range(length)]
    lst[idx] = 1
    return lst 


def run_basic_model(model,x,y,test_x):
    model.fit(x,y)
    y_pred = model.predict_proba(test_x)

    return y_pred

