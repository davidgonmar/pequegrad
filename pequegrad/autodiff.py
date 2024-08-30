from pequegrad.backend.c import grads as _grads  # noqa
from pequegrad.compile import jit as _jit  # noqa

grads = _grads


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


class fngrad:
    def __init__(self, f, wrt, externals=[], enabled=True, return_outs=False):
        self.f = f
        self.cache = dict()
        self.outsistuple = False
        self.externals = externals  # might be things like model parameters
        self.enabled = enabled
        self.graphs = []
        self.return_outs = return_outs
        self.wrt = wrt

    def get_externals(self):
        return self.externals() if callable(self.externals) else self.externals

    def __call__(self, *args):
        f = self.f
        inpshapes = tuple(tuple((tuple(x.shape), x.dtype)) for x in flatten_tree(args))
        if self.cache.get(inpshapes) is None:
            outs = f(*args)

            outs = flatten_tree(outs)
            assert len(outs) == 1, "Only one output supported"
            gs = []
            out = outs[0]
            gs.extend(grads(self.wrt, out))
            inputs = flatten_tree(args)
            assert all(
                isinstance(x, Tensor) for x in inputs
            ), "Only Tensors are supported. Functions must be pure, got {}".format(
                [type(x) for x in inputs]
            )
            outlen = len(outs)
            outs, inps = clone_graph(
                list(outs) + list(gs), list(inputs) + list(self.get_externals())
            )

            gs = outs[outlen:]
            outs = outs[:outlen]

            c = {"grads": gs, "inps": inps}

            if self.return_outs:
                c["outs"] = outs
                self.example_outs = outs

            self.cache[inpshapes] = c

            self.example_grads = gs

        # now clone c and feed data
        allouts = []
        outoffset = 0
        if self.return_outs:
            allouts.extend(self.cache[inpshapes]["outs"])
            outoffset = len(self.cache[inpshapes]["outs"])

        allouts.extend(self.cache[inpshapes]["grads"])

        allouts, inps = clone_graph(allouts, self.cache[inpshapes]["inps"])

        outs = allouts[:outoffset]
        gs = allouts[outoffset:]

        i = 0
        args = flatten_tree(args)
        for inp, arg in zip(inps, args):
            inp._inplace_as_copy(arg)
            i += 1

        for inp, arg in zip(inps[i:], self.get_externals()):
            inp._inplace_as_copy(arg)

        ret = reconstruct_tree(gs, self.example_grads)

        if self.return_outs:
            ret = (reconstruct_tree(outs, self.example_outs), ret)

        return ret
