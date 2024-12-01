import os
from urllib.request import urlretrieve
import numpy as np
from typing import Tuple
from pequegrad.tensor import Tensor

# cifar-100 dataset


cifar_100_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


def _download_cifar_100(path):
    """Download CIFAR-100 dataset to path"""
    cifar_100_path = os.path.join(path, "CIFAR-100")
    if not os.path.exists(cifar_100_path):
        os.makedirs(cifar_100_path)
        urlretrieve(
            cifar_100_url, os.path.join(cifar_100_path, "cifar-100-python.tar.gz")
        )


def get_cifar_100_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get CIFAR-100 dataset"""
    _download_cifar_100(DATA_PATH)
    cifar_100_path = os.path.join(DATA_PATH, "CIFAR-100")

    # Extract files
    import tarfile

    with tarfile.open(
        os.path.join(cifar_100_path, "cifar-100-python.tar.gz"), "r:gz"
    ) as f:
        f.extractall(cifar_100_path)

    # Load data
    import pickle

    with open(os.path.join(cifar_100_path, "cifar-100-python", "train"), "rb") as f:
        train_data = pickle.load(f, encoding="bytes")
    with open(os.path.join(cifar_100_path, "cifar-100-python", "test"), "rb") as f:
        test_data = pickle.load(f, encoding="bytes")

    X_train = (
        train_data[b"data"]
        .reshape(-1, 3, 32, 32)
        .transpose(0, 2, 3, 1)
        .astype(np.float32)
    )
    y_train = np.array(train_data[b"fine_labels"]).astype(np.int32)
    X_test = (
        test_data[b"data"]
        .reshape(-1, 3, 32, 32)
        .transpose(0, 2, 3, 1)
        .astype(np.float32)
    )
    y_test = np.array(test_data[b"fine_labels"]).astype(np.int32)

    return X_train, y_train, X_test, y_test


class CIFAR100Dataset:
    def __init__(self, tensor_device=None, train=True, transform=None):
        self.tensor_device = tensor_device
        self.train = train
        self.transforms = transform

        X_train, y_train, X_test, y_test = get_cifar_100_dataset()

        if self.train:
            self.X = Tensor(X_train)
            self.y = Tensor(y_train)
        else:
            self.X = Tensor(X_test)
            self.y = Tensor(y_test)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transforms:
            x = self.transforms(x)

        return x, y

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
