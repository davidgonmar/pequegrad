import os
from urllib.request import urlretrieve
import gzip
import numpy as np
from pequegrad.tensor import Tensor
from pequegrad.optim import Adam
from pequegrad.modules import Linear, Conv2d, Module
from pequegrad.context import no_grad
import argparse
import time

mnist_url = "http://yann.lecun.com/exdb/mnist/"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

np.random.seed(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pequegrad MNIST Example convnet")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for computations")
    args = parser.parse_args()
    CUDA = args.cuda
    print(f"Using {'CUDA' if CUDA else 'CPU'} for computations")

    class ConvNet(Module):
        def __init__(self):
            # input size = 28x28
            self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3).to(
                "cuda" if CUDA else "np"
            )
            self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=3).to(
                "cuda" if CUDA else "np"
            )
            self.fc1 = Linear(16 * 5 * 5, 10).to("cuda" if CUDA else "np")

        def forward(self, input):
            input = input.reshape((-1, 1, 28, 28))  # shape: (28, 28)
            input = (
                self.conv1.forward(input).relu().max_pool2d(kernel_size=(2, 2))
            )  # shape: (28, 28) -> (26, 26) -> (13, 13)
            input = (
                self.conv2.forward(input).relu().max_pool2d(kernel_size=(2, 2))
            )  # shape: (13, 13) -> (11, 11) -> (5, 5)
            input = input.reshape((-1, 16 * 5 * 5))
            return self.fc1.forward(input)

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
            X_train = (
                np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
                / 255.0
            )
        with gzip.open(
            os.path.join(DATA_PATH, "MNIST", "train-labels-idx1-ubyte.gz"), "rb"
        ) as f:
            y_train = np.frombuffer(f.read(), np.uint8, offset=8)
        with gzip.open(
            os.path.join(DATA_PATH, "MNIST", "t10k-images-idx3-ubyte.gz"), "rb"
        ) as f:
            X_test = (
                np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
                / 255.0
            )
        with gzip.open(
            os.path.join(DATA_PATH, "MNIST", "t10k-labels-idx1-ubyte.gz"), "rb"
        ) as f:
            y_test = np.frombuffer(f.read(), np.uint8, offset=8)

        return X_train, y_train, X_test, y_test

    def train(model, X_train, Y_train, X_test, Y_test, epochs=13, batch_size=512):
        start = time.time()
        # weights of the network printed
        optim = Adam(model.parameters(), lr=0.033)
        for epoch in range(epochs):
            indices = np.random.choice(len(X_train), batch_size)
            batch_X = Tensor(X_train[indices], storage="cuda" if CUDA else "np")
            batch_Y = Tensor(Y_train[indices], storage="cuda" if CUDA else "np")

            # Forward pass
            prediction = model.forward(batch_X)

            # Compute loss and backpropagate
            loss = prediction.cross_entropy_loss_indices(
                batch_Y,
            )

            loss.backward()

            # Update the weights
            optim.step()
            print(f"Epoch {epoch} | Loss {loss.numpy()}", end="\r")

            if epoch % 10 == 0:
                model.save("conv_mnist_model.pkl")
                print("Model saved")
        with no_grad():
            # Evaluate the model
            correct = 0
            for i in range(0, len(X_test), batch_size):
                end_idx = min(i + batch_size, len(X_test))
                batch_X = Tensor(X_test[i:end_idx], storage="cuda" if CUDA else "np")
                batch_Y = Tensor(Y_test[i:end_idx], storage="cuda" if CUDA else "np")

                prediction = model.forward(batch_X)
                correct += (
                    np.argmax(prediction.numpy(), axis=1) == batch_Y.numpy()
                ).sum()

        print(f"Test accuracy: {correct / len(X_test)}")
        print("Got {} / {} correct!".format(correct, len(X_test)))
        print(f"Time taken: {time.time() - start:.2f}s")

        model.save("conv_mnist_model.pkl")

    X_train, y_train, X_test, y_test = get_dataset()

    mlp = ConvNet()
    train(mlp, X_train, y_train, X_test, y_test)
