from .dataset import Dataset
import numpy as np


def get_indices_sequential(bs: int, batch_size: int, curr_step: int) -> np.ndarray:
    return np.arange(curr_step * bs, (curr_step + 1) * bs) % batch_size

def get_indices_random(bs: int, batch_size: int, curr_step: int) -> np.ndarray:
    return np.random.randint(0, batch_size, bs)

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = get_indices_random if shuffle else get_indices_sequential

    def __iter__(self):
        total_iters = len(self.dataset) // self.batch_size
        for i in range(total_iters):
            indices = self.sampler(self.batch_size, len(self.dataset), i)
            yield self.dataset[indices]

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx: int):
        return self.dataset[idx * self.batch_size : (idx + 1) * self.batch_size]