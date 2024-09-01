from pequegrad.backend.c import grads  # noqa
from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from .pytree import tree_flatten, tree_unflatten
from .utils import get_cache_key, extract_input_tensors, bridge_args_to_lazy_fn
from .lazyfn import LazyFunction
import itertools


def ndindex(shape):
    if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError("Shape must be a tuple of integers")
    return itertools.product(*(range(dim) for dim in shape))


def jacrev(out, wrt):
    # we can compute the jacobian by computing the gradient of each element of the output
    # that can be done by computing a vjp with v = e_i where e_i is the i-th unit vector
    jacs = []
    wrtorig = wrt
    if isinstance(wrt, Tensor):
        wrt = [wrt]
    for w in wrt:
        jac = Tensor.zeros((*out.shape, *w.shape), device=out.device)
        for i in ndindex(tuple(out.shape)):
            v = Tensor.zeros(out.shape, device=out.device)
            val = Tensor.ones([], device=out.device).astype(out.dtype)
            v = v.assign_at(val, i)
            g = grads([w], out, v)[0]
            jac = jac.assign_at(g, i)
        jacs.append(jac)

    return jacs if isinstance(wrtorig, list) else jacs[0]


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


class fnjacobian(LazyFunction):
    def __init__(self, f, wrt, enabled=True):
        self.f = f
        self.cache = dict()
        self.outsistuple = False
        self.enabled = enabled
        self.graphs = []
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
            wrt = []
            for w in self.wrt:
                if isinstance(w, Tensor):
                    wrt.append(w)
                elif isinstance(w, int):
                    wrt.append(inputs[w])

            jac = jacrev(out, wrt)
            jac, inptensors = clone_graph(jac, inptensors)

            c = {"jac": jac, "inps": inptensors}

            self.cache[cache_key] = c

            self.inputs_pytree = inputs_pytree

        # now clone c and feed data
        jac, inps = clone_graph(
            self.cache[cache_key]["jac"], self.cache[cache_key]["inps"]
        )

        args, args_pytree = tree_flatten(args)
        args = [x for x in args if isinstance(x, Tensor)]

        bridge_args_to_lazy_fn(inps, args)

        return jac

    def print_trace(self):
        # traverses the graph and prints a function like representation
        name_map = {}
        curr = 0

        def n(x):
            nonlocal curr
            if x not in name_map:
                name_map[x] = f"v{curr}"
                curr += 1
            return name_map[x]

        cache = self.cache
        outs = []
        if cache[next(iter(cache))].get("outs") is not None:
            outs = cache[next(iter(cache))]["outs"]
        if cache[next(iter(cache))].get("jac") is not None:
            outs = cache[next(iter(cache))]["jac"]
        inps = cache[next(iter(cache))]["inps"]

        def dtype_name(x):
            if x.dtype == dt.float32:
                return "f32"
            if x.dtype == dt.float64:
                return "f64"
            if x.dtype == dt.int32:
                return "i32"

        def repr_tensor(x):
            return f"{n(x)}: {dtype_name(x)}[{', '.join([str(y) for y in x.shape])}]"

        strres = "f(" + ", ".join([repr_tensor(x) for x in inps]) + "){" + "\n"

        body = []
        visited = set(x for x in inps)

        def recurse(x):
            nonlocal body
            if x not in visited:
                for child in x.children():
                    recurse(child)
                body.append(
                    f"{repr_tensor(x)} = {x.ad_context()}({', '.join([n(y) for y in x.children()])})"
                )
                visited.add(x)

        for out in outs:
            recurse(out)
        strres += "  " + "\n  ".join(body) + "\n"

        # return statement
        strres += f"  return {', '.join([n(x) for x in outs])}" + "\n"
        strres += "}"

        print(strres)
