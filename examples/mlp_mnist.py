from pequegrad.extra.mnist import get_mnist_dataset
import numpy as np
from pequegrad.tensor import Tensor
from pequegrad.optim import SGD
from pequegrad.modules import Linear, Module
from pequegrad.context import no_grad
import argparse
import cProfile
import pstats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    args = parser.parse_args()
    USE_CUDA = args.cuda

    print("Using CUDA" if USE_CUDA else "Using CPU")

    class MLP(Module):
        def __init__(self):
            self.fc1 = Linear(784, 200).to("cuda" if USE_CUDA else "np")
            self.fc2 = Linear(200, 10).to("cuda" if USE_CUDA else "np")

        def forward(self, input):
            input = self.fc1.forward(input).relu()
            return self.fc2.forward(input)

        def parameters(self):
            return self.fc1.parameters() + self.fc2.parameters()

        def zero_grad(self):
            self.fc1.zero_grad()
            self.fc2.zero_grad()

    def train(model, X_train, Y_train, X_test, Y_test, epochs=8, batch_size=2048):
        # weights of the network printed
        optim = SGD(model.parameters(), lr=0.004, weight_decay=0, momentum=0.05)
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                # Determine the end index of the current batch
                end_idx = min(i + batch_size, len(X_train))
                batch_X = Tensor(
                    X_train[i:end_idx], storage="cuda" if USE_CUDA else "np"
                )
                batch_Y = Tensor(
                    Y_train[i:end_idx], storage="cuda" if USE_CUDA else "np"
                )

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
                batch_X = Tensor(
                    X_test[i:end_idx], storage="cuda" if USE_CUDA else "np"
                )
                batch_Y = Tensor(
                    Y_test[i:end_idx], storage="cuda" if USE_CUDA else "np"
                )

                prediction = model.forward(batch_X)
                correct += (
                    np.argmax(prediction.numpy(), axis=1) == batch_Y.numpy()
                ).sum()

        print(f"Test accuracy: {correct / len(X_test)}")
        print("Got {} / {} correct!".format(correct, len(X_test)))

    X_train, y_train, X_test, y_test = get_mnist_dataset()

    profiler = cProfile.Profile()

    profiler.enable()
    mlp = MLP()
    mlp.zero_grad()
    train(mlp, X_train, y_train, X_test, y_test)
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats("cumulative")

    stats.print_stats()
