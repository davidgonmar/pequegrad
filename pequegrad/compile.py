from pequegrad.backend.c import compile, clone_graph, Tensor  # noqa


def jit(f):
    c = None

    def wrapper(*args):
        nonlocal c
        if c is None:
            outs = f(*args)
            inputs = args
            assert all(
                isinstance(x, Tensor) for x in inputs
            ), "Only Tensors are supported. Functions must be pure."
            cloned_graph = clone_graph(outs, inputs)
            compile(cloned_graph)
        return f(*args)

    return wrapper
