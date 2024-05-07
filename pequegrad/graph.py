from pequegrad.tensor import Tensor  # noqa
from pequegrad.backend.c import ComputeGraph  # noqa


def from_fn(fn):
    """
    Create a ComputeGraph from a function.
    """

    _fn = fn
    # returns a mock ComputeGraph. Once a first call is made, it will be replaced by the real ComputeGraph
    # ComputeGraph can only be generated from 'outputs', an array of the outputs of the function
    # But we instead want to generate it from the function itself, so we do it 'lazily'
    # And we return the actual ComputeGraph object only when the first call is made
    graph = None

    def _wrapper(*args):
        nonlocal graph
        if graph is None:
            outputs = _fn(*args)
            graph = ComputeGraph.from_outputs(
                outputs if isinstance(outputs, (list, tuple)) else [outputs], args
            )
        return graph.feed_data(args, True)

    return _wrapper


ComputeGraph.from_fn = from_fn
