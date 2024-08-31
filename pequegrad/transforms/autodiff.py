from pequegrad.backend.c import grads as _grads  # noqa
from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from .pytree import tree_flatten, tree_unflatten
from .utils import get_cache_key, extract_input_tensors, bridge_args_to_lazy_fn
from .lazyfn import LazyFunction

grads = _grads


class fngrad(LazyFunction):
    def __init__(self, f, wrt, enabled=True, return_outs=False):
        self.f = f
        self.cache = dict()
        self.outsistuple = False
        self.enabled = enabled
        self.graphs = []
        self.return_outs = return_outs
        self.wrt = wrt

    def __call__(self, *args):
        f = self.f
        inputs, inputs_pytree = tree_flatten(args)
        inptensors = extract_input_tensors(inputs)
        cache_key = get_cache_key(inputs)
        if self.cache.get(cache_key) is None:
            outs = f(*args)
            outs, outs_pytree = tree_flatten(outs)
            assert len(outs) == 1, "Only one output supported"
            out = outs[0]
            gs = grads(self.wrt, out)
            outlen = len(outs)
            outs, inptensors = clone_graph(list(outs) + list(gs), list(inptensors))
            gs = outs[outlen:]
            outs = outs[:outlen]

            c = {"grads": gs, "inps": inptensors}

            if self.return_outs:
                c["outs"] = outs
                self.outs_pytree = outs_pytree

            self.cache[cache_key] = c

            self.inputs_pytree = inputs_pytree

        # now clone c and feed data
        allouts = []
        outoffset = 0
        if self.return_outs:
            allouts.extend(self.cache[cache_key]["outs"])
            outoffset = len(self.cache[cache_key]["outs"])

        allouts.extend(self.cache[cache_key]["grads"])

        allouts, inps = clone_graph(allouts, self.cache[cache_key]["inps"])

        outs = allouts[:outoffset]
        gs = allouts[outoffset:]

        args, args_pytree = tree_flatten(args)
        args = [x for x in args if isinstance(x, Tensor)]

        bridge_args_to_lazy_fn(inps, args)

        return (
            gs if not self.return_outs else (tree_unflatten(self.outs_pytree, outs), gs)
        )
