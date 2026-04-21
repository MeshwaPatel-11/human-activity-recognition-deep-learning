import torch
from torch.utils.data import Dataset

class HARWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # UCI HAR labels are 1..6 -> convert to 0..5 for PyTorch
        self.y = torch.tensor(y - 1, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]