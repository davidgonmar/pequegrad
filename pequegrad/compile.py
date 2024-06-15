from pequegrad.backend.c import compile, clone_graph, Tensor, grads  # noqa

_compile = True


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
            # get grads before compiling
            self.outsistuple = isinstance(outs, (tuple, list))
            outs = outs if isinstance(outs, (tuple, list)) else [outs]
            inputs = args
            assert all(
                isinstance(x, Tensor) for x in inputs
            ), "Only Tensors are supported. Functions must be pure."
            outs, inps, _ = clone_graph(
                outs, list(inputs) + list(self.get_externals()), []
            )

            c = {"outs": outs, "grads": grads, "inps": inps}

            self.cache = c

            for out in outs:
                compile(out)

        # now clone c and feed data
        outs, inps, _ = clone_graph(
            self.cache["outs"], self.cache["inps"], []
        )  # inps already contains externals

        i = 0
        for inp, arg in zip(inps, args):
            inp.assign(arg)
            i += 1

        for inp, arg in zip(inps[i:], self.get_externals()):
            inp.assign(arg)

        return outs if self.outsistuple else outs[0]

    def get_grads_graph(self, out, wrt):
        grad_graph = grads(wrt, out)
        return grad_graph
