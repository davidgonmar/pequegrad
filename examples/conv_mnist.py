from pequegrad.extra.mnist import MNISTDataset
import numpy as np
from pequegrad.optim import Adam
from pequegrad.modules import (
    Linear,
    Conv2d,
    StatefulModule,
    Sequential,
    Reshape,
    ReLU,
    MaxPool2d,
)
from pequegrad.context import no_grad
import argparse
import time
from pequegrad.backend.c import device, Tensor, grads
from pequegrad.data.dataloader import DataLoader
from pequegrad.compile import jit

np.random.seed(0)

dev = None


class ConvNet(StatefulModule):
    def __init__(self):
        # input size = 28x28
        self.convstack = Sequential(
            Reshape((-1, 1, 28, 28)),  # shape: (28, 28)
            Conv2d(
                in_channels=1, out_channels=8, kernel_size=3
            ),  # shape: (28, 28) -> (26, 26)
            ReLU(),  # shape: (26, 26)
            MaxPool2d(kernel_size=(2, 2)),  # shape: (26, 26) -> (13, 13)
            Conv2d(
                in_channels=8, out_channels=16, kernel_size=3
            ),  # shape: (13, 13) -> (11, 11)
            ReLU(),  # shape: (11, 11)
            MaxPool2d(kernel_size=(2, 2)),  # shape: (11, 11) -> (5, 5)
            Reshape((-1, 16 * 5 * 5)),  # shape: (5, 5) -> (16 * 5 * 5)
        )

        self.fc1 = Linear(16 * 5 * 5, 10)

    def forward(self, input):
        return self.fc1(self.convstack(input))


def test_model(model, ds):
    with no_grad():
        batch_size = 512
        # Evaluate the model
        correct = 0
        total = 0
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        for x, y in dl:
            y_pred = model.forward(x).numpy()
            y_pred = np.argmax(y_pred, axis=1)
            correct += np.sum(y_pred == y.numpy())
            total += y.shape[0]
        return correct, total


def train(model, ds, epochs=13, batch_size=512):
    # weights of the network printed
    optim = Adam(model.parameters(), lr=0.033)

    def training_step(batch_X, batch_Y):
        prediction = model.forward(batch_X)
        loss = prediction.cross_entropy_loss_probs(batch_Y)
        g = grads(model.parameters(), loss)
        return [loss] + g

    training_step = jit(training_step, externals=model.parameters())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    start = None
    i = 0
    for x, y in loader:
        if i == 1:
            start = time.time()
        # Forward pass
        y = Tensor.one_hot(10, y, device=dev)
        outs = training_step(x, y)
        loss = outs[0]
        g = outs[1:]
        optim.step(g)
        print(
            f"Epoch {i} | Loss {loss.numpy()}",
            end="\r" if i < epochs - 1 else "\n",
        )
        i += 1
        if i >= epochs:
            break

    end = time.time()
    print(f"Training time: {end - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pequegrad MNIST Example convnet")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for computations")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Mode: train or eval",
        choices=["train", "eval"],
    )
    args = parser.parse_args()
    CUDA = args.cuda
    MODE = args.mode
    model = ConvNet()
    if CUDA:
        dev = device.cuda

        print("Using CUDA for computations")
    else:
        dev = device.cpu
        print("Using CPU for computations")

    model.to(dev)
    if MODE == "eval":
        model.load("conv_mnist_model.pkl")
        print("Model loaded from conv_mnist_model.pkl")
        ds = MNISTDataset(device=dev, train=False)
        correct, total = test_model(model, ds)
        print(f"Test accuracy: {correct / total}")

    else:
        ds = MNISTDataset(device=dev, train=True)
        print("Training...")
        train(model, ds, epochs=13)

        correct, total = test_model(model, ds)
        print(f"Train accuracy: {correct / total}")
        print("Saving model...")
        model.save("conv_mnist_model.pkl")
        print("Model saved to conv_mnist_model.pkl")
