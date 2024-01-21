import os
from urllib.request import urlretrieve
import gzip
import numpy as np
from pequegrad.tensor import Tensor
from pequegrad.optim import SGD

mnist_url = "http://yann.lecun.com/exdb/mnist/"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class LinearLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # kaiming initialization
        self.weights = Tensor(
            np.random.normal(
                0, np.sqrt(1 / input_size), (output_size, input_size)
            ).tolist(),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(output_size).tolist(), requires_grad=True)

    def forward(self, input):
        return (input @ self.weights.transpose()) + self.bias

    def parameters(self):
        return [self.weights, self.bias]

    def zero_grad(self):
        self.weights.zero_grad()
        self.bias.zero_grad()

    def backward(self, gradient):
        self.weights.backward(gradient)
        self.bias.backward(gradient)


class MLP:
    def __init__(self):
        self.LL1 = LinearLayer(784, 200)
        self.LL2 = LinearLayer(200, 10)

    def forward(self, input):
        input = self.LL1.forward(input).relu()
        return self.LL2.forward(input)

    def parameters(self):
        return self.LL1.parameters() + self.LL2.parameters()

    def zero_grad(self):
        self.LL1.zero_grad()
        self.LL2.zero_grad()


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


def train(model, X_train, Y_train, X_test, Y_test, epochs=15, batch_size=32):
    # weights of the network printed
    optim = SGD(model.parameters(), lr=0.001, weight_decay=0)
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            # Determine the end index of the current batch
            end_idx = min(i + batch_size, len(X_train))
            batch_X = Tensor(X_train[i:end_idx])
            batch_Y = Y_train[i:end_idx]

            # Forward pass
            prediction = model.forward(batch_X)

            # Convert batch_Y to one-hot encoding (just passing the value is not supported yet)
            batch_Y_one_hot = np.zeros((end_idx - i, 10))
            batch_Y_one_hot[np.arange(end_idx - i), batch_Y] = 1

            # Compute loss and backpropagate
            loss = prediction.cross_entropy_loss(
                Tensor(batch_Y_one_hot, requires_grad=True)
            )

            loss.backward()

            # Update the weights
            optim.step()

        print(f"Epoch {epoch} | Loss {loss.data}")

    # Evaluate the model
    correct = 0
    for i in range(0, len(X_test), batch_size):
        end_idx = min(i + batch_size, len(X_test))
        batch_X = Tensor(X_test[i:end_idx])
        batch_Y = Y_test[i:end_idx]

        prediction = model.forward(batch_X)
        correct += (np.argmax(prediction.data, axis=1) == batch_Y).sum()

    print(f"Test accuracy: {correct / len(X_test)}")
    print("Got {} / {} correct!".format(correct, len(X_test)))


X_train, y_train, X_test, y_test = get_dataset()

mlp = MLP()
mlp.zero_grad()
train(mlp, X_train, y_train, X_test, y_test)
