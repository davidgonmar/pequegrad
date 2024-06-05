import os
import gzip
from urllib.request import urlretrieve
import numpy as np
from typing import Tuple
from pequegrad.backend.c import Tensor


# cifar-100 dataset


cifar_100_url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


def _download_cifar_100(path):
    """Download CIFAR-100 dataset to path"""
    cifar_100_path = os.path.join(path, "CIFAR-100")
    if not os.path.exists(cifar_100_path):
        os.makedirs(cifar_100_path)
    urlretrieve(cifar_100_url, os.path.join(cifar_100_path, "cifar-100-python.tar.gz"))


def get_cifar_100_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get CIFAR-100 dataset from path"""

    # first, check if the dataset exists in path
    if not os.path.exists(os.path.join(DATA_PATH, "CIFAR-100")):
        # if not, download the dataset
        _download_cifar_100(DATA_PATH)

    # load the dataset
    with gzip.open(
        os.path.join(DATA_PATH, "CIFAR-100", "cifar-100-python.tar.gz"), "rb"
    ) as f:
        X_train = (
            np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 32 * 32 * 3)
            / 255.0
        ).astype(np.float32)
    with gzip.open(
        os.path.join(DATA_PATH, "CIFAR-100", "cifar-100-python.tar.gz"), "rb"
    ) as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)

    return X_train, y_train, X_train, y_train


class CIFAR100Dataset:
    def __init__(self, tensor_device=None):
        self.tensor_device = tensor_device
        self.X_train, self.y_train, self.X_test, self.y_test = get_cifar_100_dataset(
            tensor_device=self.tensor_device
        )

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        X = self.X_train[idx]
        y = self.y_train[idx]
        return Tensor(X, device=self.tensor_device), Tensor(
            y, device=self.tensor_device
        )

    def test(self):
        return Tensor(self.X_test, device=self.tensor_device), Tensor(
            self.y_test, device=self.tensor_device
        )
