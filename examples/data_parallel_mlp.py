from pequegrad.extra.mnist import MNISTDataset
import numpy as np
from pequegrad.modules import Linear, StatefulModule
from pequegrad.context import no_grad
import time
from pequegrad.optim import Adam, SGD, JittedAdam  # noqa
from pequegrad.data.dataloader import DataLoader
from pequegrad.tensor import Tensor
from pequegrad import fngrad
import pequegrad as pg

np.random.seed(0)


class MLP(StatefulModule):
    def __init__(self):
        self.fc1 = Linear(784, 284)
        self.fc2 = Linear(284, 10)

    def forward(self, input):
        x = self.fc1(input).relu()
        x = self.fc2(x)
        return x


def train(model, ds, epochs=2, batch_size=8192):
    NUM_DEVICES = 8
    pg.device.force_emulated_devices(
        NUM_DEVICES, "cuda"
    )  # force 8 emulated cuda devices
    model.to("cuda:0")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def get_loss(batch_X, batch_Y, model):
        prediction = model.forward(batch_X)
        return prediction.cross_entropy_loss_probs(batch_Y)

    loss_and_grads = fngrad(get_loss, wrt=[2], return_outs=True)

    loss_and_grad_multidevice = pg.pmap(
        loss_and_grads,
        devices=[f"cuda:{i}" for i in range(NUM_DEVICES)],  # 8 devices
        argnum_opts=[0, 0, None],  # None means replicate
    )  # should match the above function
    # loss_and_grad_multidevice = pg.jit(loss_and_grad_multidevice)  # jit the function
    i = 0
    optim = Adam(model.parameters(), lr=0.021)
    for epoch in range(epochs):
        for x, y in loader:
            if i == 1:
                start = time.time()
            batch_y_onehot = Tensor.one_hot(10, y, device="cuda:0").reshape(
                (NUM_DEVICES, -1, 10)
            )  # 8 for 8 devices
            x = x.to("cuda:0").reshape((NUM_DEVICES, -1, 784))  # 8 for 8 devices
            # subdivide the batch into smaller batches to shard the computation among the available devict es
            loss, grads = loss_and_grad_multidevice(x, batch_y_onehot, model)
            # loss_and_grad_multidevice.print_trace()
            optim.step(grads)
            print(f"Step {i} | Loss {loss.numpy()}")

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

    model = model.to("cuda:0")
    start = None
    i = 0
    for xx in range(1):
        for x, y in loader:
            with no_grad():
                if i == 1:  # start time after first batch
                    start = time.time()
                x = x.to("cuda:0")
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
    mlp = MLP()
    print("Training the model")
    train_ds = MNISTDataset(device="cuda")
    test_ds = MNISTDataset(device="cuda", train=False)
    mlp = train(mlp, train_ds)
    correct, total = test_model(mlp, test_ds)
