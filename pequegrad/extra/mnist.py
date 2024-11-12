import os
import gzip
from urllib.request import urlretrieve
import numpy as np
from typing import Tuple
from pequegrad.backend.c import Tensor

mnist_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


def _download_mnist(path):
    """Download MNIST dataset to path"""
    mnist_path = os.path.join(path, "MNIST")
    if not os.path.exists(mnist_path):
        os.makedirs(mnist_path)
    for name in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        urlretrieve(mnist_url + name, os.path.join(mnist_path, name))


def get_mnist_dataset(
    tensor_device=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get MNIST dataset from path"""

    # first, check if the dataset exists in path
    if not os.path.exists(os.path.join(DATA_PATH, "MNIST")):
        # if not, download the dataset
        _download_mnist(DATA_PATH)

    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    # check if all the files are present
    for name in files:
        if not os.path.exists(os.path.join(DATA_PATH, "MNIST", name)):
            _download_mnist(DATA_PATH)
            break

    # load the dataset
    with gzip.open(
        os.path.join(DATA_PATH, "MNIST", "train-images-idx3-ubyte.gz"), "rb"
    ) as f:
        X_train = (
            np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28) / 255.0
        ).astype(np.float32)
    with gzip.open(
        os.path.join(DATA_PATH, "MNIST", "train-labels-idx1-ubyte.gz"), "rb"
    ) as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)
    with gzip.open(
        os.path.join(DATA_PATH, "MNIST", "t10k-images-idx3-ubyte.gz"), "rb"
    ) as f:
        X_test = (
            np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28) / 255.0
        ).astype(np.float32)
    with gzip.open(
        os.path.join(DATA_PATH, "MNIST", "t10k-labels-idx1-ubyte.gz"), "rb"
    ) as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)

    return (
        (X_train, y_train, X_test, y_test)
        if tensor_device is None
        else (
            Tensor(X_train, device=tensor_device),
            Tensor(y_train, device=tensor_device),
            Tensor(X_test, device=tensor_device),
            Tensor(y_test, device=tensor_device),
        )
    )


class MNISTDataset:
    def __init__(self, device, train=True):
        self.device = device
        self.train = train
        x_train, y_train, x_test, y_test = get_mnist_dataset(device)

        if train:
            self.x, self.y = x_train, y_train
            del x_test, y_test
        else:
            self.x, self.y = x_test, y_test
            del x_train, y_train

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
