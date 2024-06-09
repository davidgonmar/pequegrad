from pequegrad.backend.c import compile, clone_graph, Tensor  # noqa

_compile = True
def jit(f,):
    c = None
    outsistuple = False
    def wrapper(*args):
        nonlocal outsistuple
        nonlocal c
        if c is None:
            outs = f(*args)
            outsistuple = isinstance(outs, (tuple, list))
            outs = outs if isinstance(outs, (tuple, list)) else (outs,)
            inputs = args
            assert all(
                isinstance(x, Tensor) for x in inputs
            ), "Only Tensors are supported. Functions must be pure."
            outs, inps = clone_graph(outs, inputs)
            c = dict()
            c["outs"] = outs
            c["inps"] = inps
            if _compile:
                assert len(outs) == 1, "Only single output functions are supported"
                compile(outs[0])

        # now clone c and feed data
        outs, inps = clone_graph(c["outs"], c["inps"])

        for inp, arg in zip(inps, args):
            inp.assign(arg)
        
        #print(inps, outs)
        return outs if outsistuple else outs[0]
    return wrapper

