from pequegrad.extra.mnist import get_mnist_dataset
import numpy as np
from pequegrad.modules import Linear, StatefulModule
from pequegrad.context import no_grad
import argparse
import time
from pequegrad.backend.c import device, grads
from pequegrad.optim import Adam

np.random.seed(0)

model_path = "mlp_mnist_model.pkl"

device = device.cpu


class MLP(StatefulModule):
    def __init__(self):
        self.fc1 = Linear(784, 200)
        self.fc2 = Linear(200, 10)

    def forward(self, input):
        input = self.fc1.forward(input).relu()
        return self.fc2.forward(input)


def train(model, X_train, Y_train, epochs=13, batch_size=4096):
    # weights of the network printed
    optim = Adam(model.parameters(), lr=0.021)
    for epoch in range(epochs):
        indices = np.random.choice(len(X_train), batch_size)
        batch_X = X_train[indices]
        batch_Y = Y_train[indices]

        # Forward pass
        prediction = model.forward(batch_X)
        # Compute loss and backpropagate
        loss = prediction.cross_entropy_loss_indices(batch_Y)
        loss.eval()
        g = grads(model.parameters(), loss)

        # Update the weights
        optim.step(g)

        print(f"Epoch {epoch} | Loss {loss.numpy()}")

    return model


def test_model(model, X_test, Y_test):
    with no_grad():
        batch_size = 2048
        # Evaluate the model
        correct = 0
        for i in range(0, len(X_test), batch_size):
            end_idx = min(i + batch_size, len(X_test))
            batch_X = X_test[i:end_idx]
            batch_Y = Y_test[i:end_idx]
            prediction = model.forward(batch_X)
            correct += (np.argmax(prediction.numpy(), axis=1) == batch_Y.numpy()).sum()
        return correct, len(X_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode: train or test",
    )
    args = parser.parse_args()
    USE_CUDA = args.cuda
    MODE = args.mode
    mlp = MLP()
    if USE_CUDA:
        print("Using CUDA")
        mlp.to(device.cuda)
        device = device.cuda
    else:
        mlp.to(device.cpu)
        print("Using CPU")
        device = device.cpu
    X_train, y_train, X_test, y_test = get_mnist_dataset(tensor_device=device)
    if MODE == "train":
        print("Training the model")
        start = time.time()
        mlp = train(mlp, X_train, y_train)
        print(f"Training took {time.time() - start:.2f} seconds")
        mlp.save(model_path)

        print("Model saved to", model_path)
        print("Testing the model")
        correct, total = test_model(mlp, X_test, y_test)
        print(f"Test accuracy: {correct / total:.3f}")
    else:
        print("Evaluating the model")
        mlp.load(model_path)
        print("Model loaded from", model_path)
        print("Testing the model")
        correct, total = test_model(mlp, X_test, y_test)
        print(f"Test accuracy: {correct / total:.3f}")
