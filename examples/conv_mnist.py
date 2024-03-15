from pequegrad.extra.mnist import get_mnist_dataset
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

np.random.seed(0)


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


def test_model(model, X_test, Y_test):
    with no_grad():
        batch_size = 512
        # Evaluate the model
        correct = 0
        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            batch_X = X_test[i:end_idx]
            batch_Y = Y_test[i:end_idx]
            prediction = model.forward(batch_X)

            correct += (np.argmax(prediction.numpy(), axis=1) == batch_Y.numpy()).sum()

        return correct, len(X_test)


def train(model, X_train, Y_train, epochs=13, batch_size=512):
    # pro = cProfile.Profile()
    # pro.enable()
    # weights of the network printed
    optim = Adam(model.parameters(), lr=0.033)
    for epoch in range(epochs):
        indices = np.random.choice(len(X_train), batch_size)
        batch_X = X_train[indices]
        batch_Y = Y_train[indices]
        # Forward pass
        prediction = model.forward(batch_X)
        # Compute loss and backpropagate
        loss = prediction.cross_entropy_loss_indices(batch_Y)
        loss.backward()
        # Update the weights
        optim.step()
        print(
            f"Epoch {epoch} | Loss {loss.numpy()}",
            end="\r" if epoch < epochs - 1 else "\n",
        )

    # pro.disable()
    # pro.print_stats(sort="time")


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
        model.to("cuda")
        print("Using CUDA for computations")
    else:
        model.to("np")
        print("Using CPU for computations")

    if MODE == "eval":
        model.load("conv_mnist_model.pkl")
        print("Model loaded from conv_mnist_model.pkl")
        X_train, y_train, X_test, y_test = get_mnist_dataset(
            backend="cuda" if CUDA else "np"
        )
        correct, total = test_model(model, X_test, y_test)
        print(f"Test accuracy: {correct / total}")

    else:
        X_train, y_train, X_test, y_test = get_mnist_dataset(
            backend="cuda" if CUDA else "np"
        )
        start = time.time()
        train(model, X_train, y_train, epochs=13, batch_size=512)
        print(f"Time taken to train: {(time.time() - start):.2f}s")

        print("Evaluating model...", end="\r")
        correct, total = test_model(model, X_test, y_test)
        print(f"Test accuracy: {correct / total}")

        print("Saving model...")
        model.save("conv_mnist_model.pkl")
        print("Model saved to conv_mnist_model.pkl")
