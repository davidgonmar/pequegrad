from pequegrad.extra.mnist import MNISTDataset
import numpy as np
from pequegrad.modules import Linear, StatefulModule, apply_to_module
from pequegrad.context import no_grad
import argparse
import time
from pequegrad.backend.c import device
from pequegrad.optim import adam, AdamState  # noqa
from pequegrad.data.dataloader import DataLoader
from pequegrad import fngrad, jit, amp, maybe
from .quantized_mlp import quantized
import pequegrad as pg

np.random.seed(0)

model_path = "mlp_mnist_model.pkl"
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
    use_jit = True
    do_amp = False
    allocator = "custom"

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    @maybe(jit.withargs(allocator=allocator), use_jit)
    @maybe(amp, do_amp)
    def update(optim_state, params_dict, x, y):
        def get_loss(batch_X, batch_Y, params_dict):
            prediction = apply_to_module(model, params_dict, batch_X)
            return prediction.cross_entropy_loss_probs(batch_Y)

        loss, (g,) = fngrad(get_loss, wrt=[2], return_outs=True)(x, y, params_dict)
        new_optim_state, new_params = adam(params_dict, g, optim_state)
        return new_optim_state, new_params, loss

    i = 0
    optim_state = AdamState(model)
    params_dict = model.tree_flatten()
    for x, y in loader:
        if i == 1:
            start = time.time()
        batch_y_onehot = pg.one_hot(10, y, device=device)
        optim_state, params_dict, loss = update(
            optim_state, params_dict, x, batch_y_onehot
        )
        update.print_trace()
        print(f"Step {i} | Loss {loss.numpy()}")
        if i >= epochs:
            break
        i += 1

    end = time.time()
    print(f"Training time: {end - start:.2f}s")

    return params_dict


def test_model(model, params_dict, ds):
    import time

    correct = 0
    total = 0
    loader = DataLoader(ds, batch_size=4096)

    def step(x, model, params_dict):
        return apply_to_module(model, params_dict, x)

    step = jit(amp(step), eval_outs=False)
    start = None
    i = 0
    for xx in range(1):
        for x, y in loader:
            with no_grad():
                if i == 1:  # start time after first batch
                    start = time.time()
                outputs = step(x, model, params_dict)

                correct += np.sum(outputs.numpy().argmax(1) == y.numpy())
                total += y.shape[0]
                i += 1
    end = time.time()
    print(
        f"Correct: {correct}, Total: {total}, Accuracy: {correct / total:.3f}, Time: {end - start:.5f}"
    )
    return correct, total


def test_model_quant(model, params_dict, ds):
    import time

    correct = 0
    total = 0
    loader = DataLoader(ds, batch_size=4096)

    def step(x, model, params_dict):
        return apply_to_module(model, params_dict, x)

    step = jit(quantized(step), eval_outs=False)
    start = None
    i = 0
    for xx in range(1):
        for x, y in loader:
            with no_grad():
                if i == 1:  # start time after first batch
                    start = time.time()
                outputs = step(x, model, params_dict)
                pg.viz(outputs)
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
        d = train(mlp, train_ds)
        mlp.save(model_path)

        print("Model saved to", model_path)
        print("Testing the model")
        correct, total = test_model_quant(mlp, d, test_ds)
        print(f"Test accuracy: {correct / total:.3f}")
    else:
        raise ValueError("Unknown mode")
