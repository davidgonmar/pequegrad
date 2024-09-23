import pequegrad as pg
from pequegrad.extend import Primitive


class NpAdd(Primitive):
    @staticmethod
    def dispatch(inputs):
        npret = inputs[0].numpy() + inputs[1].numpy()
        return pg.Tensor(npret).to(inputs[0].device)

    @staticmethod
    def backward(primals, cotangents, outputs):
        gout = cotangents[0]

        return gout * 2, gout * 2  # make sure this fn is being dispatched correctly


def add(a, b):
    return NpAdd.apply(a, b)


if __name__ == "__main__":
    a = pg.Tensor([1.0, 2.0, 3.0])
    b = pg.Tensor([2.0, 3.0, 4.0])
    c = add(a, b)
    g = pg.grads([a, b], c)
    print(c.numpy())
    print(g[0].numpy())
    print(g[1].numpy())
    print("Done!")
