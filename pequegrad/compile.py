from pequegrad.backend.c import compile, clone_graph, Tensor, grads  # noqa

_compile = True

class TensorJitted(Tensor):
    pass

class jit:
    def __init__(self, f, aot_grads=False):
        self.f = f
        self.aot_grads = aot_grads
        self.cache = None

    def __call__(self, *args):
        f = self.f
        get_grads_graph = self.get_grads_graph
        outsistuple = False
        if self.cache is None:
                outs = f(*args)
                # get grads before compiling
                grads = get_grads_graph(outs, args) if self.aot_grads else []
                outsistuple = isinstance(outs, (tuple, list))
                outs = outs if isinstance(outs, (tuple, list)) else [outs]
                inputs = args
                assert all(
                    isinstance(x, Tensor) for x in inputs
                ), "Only Tensors are supported. Functions must be pure."
                outs, inps = clone_graph(outs + grads, inputs)
                c = {"outs": outs, "inps": inps, "grads": grads}
                self.cache = c
                if _compile:
                    assert len(outs) - len(grads) == 1, "Only single output functions are supported, got %d" % len(outs)
                    compile(outs[0])

            # now clone c and feed data
        outs, inps = clone_graph(self.cache["outs"], self.cache["inps"])

        for inp, arg in zip(inps, args):
                inp.assign(arg)
            
        return outs if outsistuple else outs[0]

    def get_grads_graph(self, out, wrt):
        grad_graph = grads(wrt, out)
        return grad_graph