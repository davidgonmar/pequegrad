import os
from urllib.request import urlretrieve
import gzip
import numpy as np
from pequegrad.tensor import Tensor
from pequegrad.optim import SGD
from pequegrad.modules import Linear, Conv2d
from pequegrad.context import no_grad

mnist_url = "http://yann.lecun.com/exdb/mnist/"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

np.random.seed(0)


class ConvNet:
    def __init__(self):
        # input size = 28x28
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc1 = Linear(16 * 5 * 5, 200)
        self.fc2 = Linear(200, 10)

    def forward(self, input):
        input = input.reshape((-1, 1, 28, 28))  # shape: (28, 28)
        input = (
            self.conv1.forward(input).relu().max_pool2d(kernel_size=(2, 2))
        )  # shape: (28, 28) -> (26, 26) -> (13, 13)
        input = (
            self.conv2.forward(input).relu().max_pool2d(kernel_size=(2, 2))
        )  # shape: (13, 13) -> (11, 11) -> (5, 5)
        input = input.reshape((-1, 16 * 5 * 5))
        input = self.fc1.forward(input).relu()
        input = self.fc2.forward(input)
        return input

    def parameters(self):
        return (
            self.fc1.parameters()
            + self.conv1.parameters()
            + self.conv2.parameters()
            + self.fc2.parameters()
        )


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


def train(model, X_train, Y_train, X_test, Y_test, epochs=250, batch_size=32):
    # weights of the network printed
    optim = SGD(model.parameters(), lr=0.0025)
    for epoch in range(epochs):
        # Randomly sample batch indices
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        batch_X = Tensor(X_train[indices])
        batch_Y = Tensor(Y_train[indices])

        # Forward pass
        prediction = model.forward(batch_X)

        # Compute loss and backpropagate
        loss = prediction.cross_entropy_loss_indices(
            batch_Y,
        )

        loss.backward()

        # Update the weights
        optim.step()
        print(
            "step {} / {}".format(epoch, epochs),
            end="\r",
        )

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

mlp = ConvNet()
train(mlp, X_train, y_train, X_test, y_test)
