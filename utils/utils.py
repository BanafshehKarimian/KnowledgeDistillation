from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, y, batch_size, shuffle=True):
        assert len(y.shape) == 1, "label array must be 1D"
        n_batches = int(len(y) / batch_size)
        self.batch_size = batch_size
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            torch.randint(0, int(1e8), size=()).item()
        for _, indices in self.skf.split(self.X, self.y):
            yield indices

    def __len__(self):
        return len(self.y) // self.batch_size