from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from pequegrad.viz import viz as v  # noqa


def flatten_tree(tree):
    if isinstance(tree, (tuple, list)):
        return sum([flatten_tree(x) for x in tree], [])
    return [tree]


def reconstruct_tree(flat, example):
    if isinstance(example, (tuple, list)):
        out = []
        i = 0
        for x in example:
            if isinstance(x, (tuple, list)):
                l = len(x)
                out.append(reconstruct_tree(flat[i : i + l], x))
                i += l
            else:
                out.append(flat[i])
                i += 1
        return tuple(out) if isinstance(example, tuple) else out
    return flat[0]


class jit:
    def __init__(self, f, externals=[], enabled=True):
        self.f = f
        self.cache = dict()
        self.outsistuple = False
        self.externals = externals  # might be things like model parameters
        self.enabled = enabled
        self.graphs = []

    def get_externals(self):
        return self.externals() if callable(self.externals) else self.externals

    def __call__(self, *args):
        if not self.enabled:
            return self.f(*args)
        f = self.f
        inpshapes = tuple(tuple((tuple(x.shape), x.dtype)) for x in flatten_tree(args))
        if self.cache.get(inpshapes) is None:
            outs = f(*args)
            self.example_outs = outs
            outs = flatten_tree(outs)
            inputs = flatten_tree(args)
            assert all(
                isinstance(x, Tensor) for x in inputs
            ), "Only Tensors are supported. Functions must be pure, got {}".format(
                [type(x) for x in inputs]
            )
            outs, inps = clone_graph(outs, list(inputs) + list(self.get_externals()))

            c = {"outs": outs, "inps": inps}

            self.cache[inpshapes] = c

            compile(outs)

        # now clone c and feed data

        outs, inps = clone_graph(
            self.cache[inpshapes]["outs"], self.cache[inpshapes]["inps"]
        )  # inps already contains externals
        i = 0
        args = flatten_tree(args)
        for inp, arg in zip(inps, args):
            inp.assign(arg)
            i += 1

        for inp, arg in zip(inps[i:], self.get_externals()):
            inp.assign(arg)

        return reconstruct_tree(outs, self.example_outs)
