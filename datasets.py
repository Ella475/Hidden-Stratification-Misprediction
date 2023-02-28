from torch.utils.data import Dataset
import pandas as pd
import torch


# Define a custom dataset class
class customDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]