from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from contextvars import ContextVar
from .pytree import tree_flatten, tree_unflatten
from .utils import get_cache_key, bridge_args_to_lazy_fn
from .lazyfn import LazyFunction

inside_jit = ContextVar("inside_jit", default=False)


class jit(LazyFunction):
    def __init__(self, f, enabled=True):
        self.f = f
        self.cache = dict()
        self.outsistuple = False
        self.enabled = enabled
        self.graphs = []

    def __call__(self, *args):
        if not self.enabled or inside_jit.get():
            # handle jitting inside other jits, just use the highest level jit
            return self.f(*args)
        token = inside_jit.set(True)
        try:
            f = self.f
            inputs, in_pytree = tree_flatten(args)
            cache_key = get_cache_key(inputs)

            if self.cache.get(cache_key) is None:
                outs = f(*args)
                outs, out_pytree = tree_flatten(outs)
                assert all(
                    isinstance(x, Tensor) for x in inputs
                ), "Only Tensors are supported. Functions must be pure, got {}".format(
                    [type(x) for x in inputs]
                )
                outs, inps = clone_graph(outs, list(inputs))

                c = {"outs": outs, "inps": inps}

                self.cache[cache_key] = c
                self.out_pytree = out_pytree
                self.in_pytree = in_pytree
                compile(outs)

            # now clone c and feed data

            outs, inps = clone_graph(
                self.cache[cache_key]["outs"], self.cache[cache_key]["inps"]
            )  # inps already contains externals
            i = 0
            args, args_pytree = tree_flatten(args)
            # make sure both are the same
            assert len(self.in_pytree) == len(args_pytree)
            bridge_args_to_lazy_fn(inps, args)

            return tree_unflatten(self.out_pytree, outs)
        finally:
            inside_jit.reset(token)


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
        outs = cache[next(iter(cache))]["outs"]
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
                body.append(f"{repr_tensor(x)} = {x.ad_context()}({', '.join([n(y) for y in x.children()])})")
                visited.add(x)
        for out in outs:
            recurse(out)
        strres += "  " + "\n  ".join(body) + "\n" 

        # return statement
        strres += f"  return {', '.join([n(x) for x in outs])}" + "\n"
        strres += "}"

        print(strres)

