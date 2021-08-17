import torch
from torch.utils import data


class FunctionBoundsDataset(data.Dataset):

    def __init__(self, data_path):
        # The dataset is reasonably small; we'll just load it all into memory
        self.X, self.Y = torch.load(data_path)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
