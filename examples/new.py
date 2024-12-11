from pequegrad.nn import BaseModule, Linear, apply_fn
from pequegrad import ops
from pequegrad.optim import AdamState, adam
import pequegrad as pg
from pequegrad.extra.mnist import MNISTDataset
from pequegrad.data.dataloader import DataLoader
import tqdm


class MyModule(BaseModule):
    def __init__(self):
        self.l1 = Linear(784, 256)
        self.l2 = Linear(256, 128)
        self.l3 = Linear(128, 10)

    def apply(self, x):
        x = self.l1(x).relu()
        x = self.l2(x).relu()
        x = self.l3(x)
        return x


m = MyModule()

params = m.init(device=pg.device.cuda())


@pg.jit
def train_step(x, y, params, state):
    x, y = x.to("cuda"), y.to("cuda")

    def loss_fn(params, x, y):
        y_pred = apply_fn(m, x, params_dict=params)
        return ops.cross_entropy_loss_probs(y_pred, ops.one_hot(10, y).to(x.device))

    loss, grads = pg.value_and_grad(loss_fn)(params, x, y)
    state, params = adam(params, grads, state)
    return loss, params, state


@pg.jit
def test_step(x, y, params):
    x, y = x.to("cuda"), y.to("cuda")
    y_pred = apply_fn(m, x, params_dict=params)
    y_pred = ops.argmax(y_pred, dim=1)
    y_pred, y = y_pred.astype("float32"), y.astype("float32")
    return ops.accuracy(y_pred, y)


adam_state = AdamState(params)


train_ds = MNISTDataset("cpu", train=True)
train_dl = DataLoader(train_ds, batch_size=4096, shuffle=True)

test_ds = MNISTDataset("cpu", train=False)
test_dl = DataLoader(test_ds, batch_size=4096, shuffle=False)

for i in range(1000):
    tqdmobj = tqdm.tqdm(train_dl, desc=f"Epoch {i}")
    for x, y in tqdmobj:
        loss, params, adam_state = train_step(x, y, params, adam_state)
        tqdmobj.set_postfix({"loss": loss.numpy()})
    acc = 0
    for x, y in test_dl:
        res = test_step(x, y, params).numpy()
        acc += res

    print(f"Test accuracy: {(acc / len(test_dl))}")
