from pequegrad.backend.c import compile, clone_graph, Tensor, grads  # noqa



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
    def __init__(self, f, externals=[], aot_grads=False):
        self.f = f
        self.aot_grads = aot_grads
        self.cache = None
        self.outsistuple = False
        self.externals = externals  # might be things like model parameters

    def get_externals(self):
        return self.externals() if callable(self.externals) else self.externals

    def __call__(self, *args):
        f = self.f

        if self.cache is None:
            outs = f(*args)
            self.example_outs = outs
            outs = flatten_tree(outs)
            inputs = flatten_tree(args)
            assert all(
                isinstance(x, Tensor) for x in inputs
            ), "Only Tensors are supported. Functions must be pure, got {}".format(
                [type(x) for x in inputs]
            )
            outs, inps = clone_graph(
                outs, list(inputs) + list(self.get_externals())
            )

            c = {"outs": outs, "inps": inps}

            self.cache = c

            for out in outs:
                compile(out)
        # now clone c and feed data
        outs, inps = clone_graph(
            self.cache["outs"], self.cache["inps"]
        )  # inps already contains externals

        i = 0
        args = flatten_tree(args)
        for inp, arg in zip(inps, args):
            inp.assign(arg)
            i += 1

        for inp, arg in zip(inps[i:], self.get_externals()):
            inp.assign(arg)

        return reconstruct_tree(outs, self.example_outs)

    def get_grads_graph(self, out, wrt):
        grad_graph = grads(wrt, out)
        return grad_graph
