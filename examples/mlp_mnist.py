from pequegrad.extra.mnist import MNISTDataset
import numpy as np
from pequegrad.modules import Linear, StatefulModule
from pequegrad.context import no_grad
import argparse
import time
from pequegrad.backend.c import device
from pequegrad.optim import Adam, SGD, JittedAdam  # noqa
from pequegrad.data.dataloader import DataLoader
from pequegrad.tensor import Tensor
from pequegrad import fngrad, jit, amp

np.random.seed(0)

model_path = "mlp_mnist_model.pkl"

device = "cpu"

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
    use_jit = False
    do_amp = False
    optcls = JittedAdam
    optim = optcls(model, lr=0.021)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def get_loss(batch_X, batch_Y, model):
        prediction = model.forward(batch_X)
        return prediction.cross_entropy_loss_probs(batch_Y)

    loss_and_grads = fngrad(get_loss, wrt=[2], return_outs=True)

    i = 0
    _amp = amp if do_amp else lambda x: x
    train_step = jit(_amp(loss_and_grads)) if use_jit else loss_and_grads

    for x, y in loader:
        if i == 1:
            start = time.time()
        batch_y_onehot = Tensor.one_hot(10, y, device=device)
        loss, g = train_step(x, batch_y_onehot, model)
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

    def step(x, model):
        return model.forward(x)

    step = jit(amp(step), eval_outs=False)
    start = None
    i = 0
    for xx in range(1):
        for x, y in loader:
            with no_grad():
                if i == 1:  # start time after first batch
                    start = time.time()
                outputs = step(x, model)

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
        mlp.to("cuda")
        device = "cuda"
    else:
        mlp.to("cpu")
        print("Using CPU")
        device = "cpu"
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
