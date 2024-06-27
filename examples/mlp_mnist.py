from pequegrad.extra.mnist import MNISTDataset
import numpy as np
from pequegrad.modules import Linear, StatefulModule
from pequegrad.context import no_grad
import argparse
import time
from pequegrad.backend.c import device, grads
from pequegrad.optim import Adam, SGD, JittedAdam  # noqa
from pequegrad.data.dataloader import DataLoader
from pequegrad.compile import jit
from pequegrad.tensor import Tensor

np.random.seed(0)

model_path = "mlp_mnist_model.pkl"

device = device.cpu

USE_GRAPH = True


class MLP(StatefulModule):
    def __init__(self):
        self.fc1 = Linear(784, 284)
        self.fc2 = Linear(284, 10)

    def forward(self, input):
        x = self.fc1(input).relu()
        x = self.fc2(x)
        return x


def train(model, ds, epochs=13, batch_size=4096):
    start = None
    # weights of the network printed
    use_jit = True
    optcls = Adam if not use_jit else JittedAdam
    optim = optcls(model.parameters(), lr=0.021)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def train_step(batch_X, batch_Y):
        prediction = model.forward(batch_X)
        loss = prediction.cross_entropy_loss_probs(batch_Y)
        g = grads(model.parameters(), loss)
        return [loss] + g

    i = 0

    train_step = (
        jit(train_step, externals=model.parameters()) if use_jit else train_step
    )

    for x, y in loader:
        if i == 1:
            start = time.time()
        batch_y_onehot = Tensor.one_hot(10, y, device=device)
        outs = train_step(x, batch_y_onehot)
        loss = outs[0]
        g = outs[1:]
        # import pequegrad.viz as viz
        # viz.viz(outs, name="outs")
        optim.step(g)

        print(f"Step {i} | Loss {loss.numpy()}")
        if i >= epochs:
            break
        i += 1

    end = time.time()
    print(f"Training time: {end - start:.2f}s")

    return model


def test_model(model, ds):
    import time

    correct = 0
    total = 0
    loader = DataLoader(ds, batch_size=4096)
    step = model.forward
    start = None
    i = 0
    for i in range(1):
        for x, y in loader:
            with no_grad():
                if i == 1:  # start time after first batch
                    start = time.time()
                outputs = step(x)
                correct += np.sum(outputs.numpy().argmax(1) == y.numpy())
                total += y.shape[0]
                i += 1
    end = time.time()
    print(
        f"Correct: {correct}, Total: {total}, Accuracy: {correct / total:.3f}, Time: {end - start:.5f}"
    )
    return correct, total


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
    train_ds = MNISTDataset(device=device)
    test_ds = MNISTDataset(device=device, train=False)

    if MODE == "train":
        print("Training the model")
        mlp = train(mlp, train_ds)
        mlp.save(model_path)

        print("Model saved to", model_path)
        print("Testing the model")
        correct, total = test_model(mlp, test_ds)
        print(f"Test accuracy: {correct / total:.3f}")
    else:
        print("Evaluating the model")
        mlp.load(model_path)
        print("Model loaded from", model_path)
        print("Testing the model")
        correct, total = test_model(mlp, test_ds)
        print(f"Test accuracy: {correct / total:.3f}")
