import os
from urllib.request import urlretrieve
import gzip
import numpy as np
from pequegrad.tensor import Tensor
from pequegrad.optim import SGD
from pequegrad.modules import Linear
from pequegrad.context import no_grad

mnist_url = "http://yann.lecun.com/exdb/mnist/"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class MLP:
    def __init__(self):
        self.fc1 = Linear(784, 200)
        self.fc2 = Linear(200, 10)

    def forward(self, input):
        input = self.fc1.forward(input).relu()
        return self.fc2.forward(input)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def zero_grad(self):
        self.fc1.zero_grad()
        self.fc2.zero_grad()


def download_mnist(path):
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


def get_dataset():
    """Get MNIST dataset from path"""

    # first, check if the dataset exists in path
    if not os.path.exists(os.path.join(DATA_PATH, "MNIST")):
        # if not, download the dataset
        download_mnist(DATA_PATH)

    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    # check if all the files are present
    for name in files:
        if not os.path.exists(os.path.join(DATA_PATH, "MNIST", name)):
            download_mnist(DATA_PATH)
            break

    # load the dataset
    with gzip.open(
        os.path.join(DATA_PATH, "MNIST", "train-images-idx3-ubyte.gz"), "rb"
    ) as f:
        X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    with gzip.open(
        os.path.join(DATA_PATH, "MNIST", "train-labels-idx1-ubyte.gz"), "rb"
    ) as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(
        os.path.join(DATA_PATH, "MNIST", "t10k-images-idx3-ubyte.gz"), "rb"
    ) as f:
        X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    with gzip.open(
        os.path.join(DATA_PATH, "MNIST", "t10k-labels-idx1-ubyte.gz"), "rb"
    ) as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    return Tensor(X_train), Tensor(y_train), Tensor(X_test), Tensor(y_test)


def train(model, X_train, Y_train, X_test, Y_test, epochs=8, batch_size=512):
    # weights of the network printed
    optim = SGD(model.parameters(), lr=0.001, weight_decay=0)
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            # Determine the end index of the current batch
            end_idx = min(i + batch_size, len(X_train))
            batch_X = Tensor(X_train[i:end_idx])
            batch_Y = Tensor(Y_train[i:end_idx])

            # Forward pass
            prediction = model.forward(batch_X)

            # Compute loss and backpropagate
            loss = prediction.cross_entropy_loss_indices(batch_Y)

            loss.backward()

            # Update the weights
            optim.step()

        print(f"Epoch {epoch} | Loss {loss.numpy()}")

    with no_grad():
        # Evaluate the model
        correct = 0
        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            batch_X = Tensor(X_test[i:end_idx])
            batch_Y = Y_test[i:end_idx]

            prediction = model.forward(batch_X)
            correct += (np.argmax(prediction.numpy(), axis=1) == batch_Y.numpy()).sum()

    print(f"Test accuracy: {correct / len(X_test)}")
    print("Got {} / {} correct!".format(correct, len(X_test)))


X_train, y_train, X_test, y_test = get_dataset()

mlp = MLP()
mlp.zero_grad()
train(mlp, X_train, y_train, X_test, y_test)
