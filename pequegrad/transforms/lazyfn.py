from pequegrad.backend.c import Tensor, clone_graph, dt
from .pytree import tree_flatten, PyTreeDef
from .utils import extract_input_tensors, get_cache_key
from typing import List, Any
from dataclasses import dataclass
from .utils import bridge_args_to_lazy_fn
from .pytree import tree_unflatten
import functools


@dataclass
class GraphTrace:
    inputs: List[Any]
    inputs_pytree: PyTreeDef
    input_tensors: List[Tensor]
    outputs: List[Tensor]
    outputs_pytree: PyTreeDef


def print_trace(trace):
    # traverses the graph and prints a function like representation
    name_map = {}
    curr = 0

    def n(x):
        if not isinstance(x, Tensor):
            return str(x)
        nonlocal curr
        if x.id not in name_map:
            name_map[x.id] = f"v{curr}"
            curr += 1
        return name_map[x.id]

    inps = trace.input_tensors
    outs = trace.outputs

    def dtype_name(x):
        if x.dtype == dt.float32:
            return "f32"
        if x.dtype == dt.float64:
            return "f64"
        if x.dtype == dt.int32:
            return "i32"
        if x.dtype == dt.float16:
            return "f16"

    def repr_tensor(x):
        return f"{n(x)}: {dtype_name(x)}[{', '.join([str(y) for y in x.shape])}]<{x.device}>"

    def non_tensors_dtype(x):
        if isinstance(x, int):
            return f"{x}: i32"
        if isinstance(x, float):
            return f"{x}: f32"
        return type(x).__name__

    def repr_arg(x):
        if isinstance(x, Tensor):
            return repr_tensor(x)
        return non_tensors_dtype(x)

    strres = "f(" + ", ".join([repr_arg(x) for x in inps]) + ") {\n"

    body = []
    visited = set(x for x in inps)

    def recurse(x):
        nonlocal body
        if x.id not in visited:
            for child in x.children():
                recurse(child)
            # if it is in the inputs, do not print it
            if x.id in [x.id for x in inps]:
                return
            body.append(
                f"{repr_tensor(x)} = {x.ad_context()}({', '.join([n(y) for y in x.children()])})"  # + "is_evaled=" + str(x.is_evaled())
            )
            visited.add(x.id)

    for out in outs:
        if not isinstance(out, Tensor):
            continue
        recurse(out)
    strres += "  " + "\n  ".join(body) + "\n"

    # return statement
    strres += f"  return {', '.join([n(x) for x in outs])}" + "\n"
    strres += "}"

    print(strres)


class Cache(dict):
    def __getitem__(self, __key: Any) -> GraphTrace:
        return super().get(__key, None)


class LazyFunction:
    cache: Cache

    def __init__(self, f, assume_static_argnums=None):
        self.f = f
        self.cache = Cache()
        self.assume_static_argnums = assume_static_argnums or []

    @classmethod
    def withargs(cls, *args, **kwargs):
        return functools.partial(cls, *args, **kwargs)

    def get_last_trace(self):
        assert len(self.cache) > 0, "No cache entries"
        last_key = list(self.cache.keys())[-1]
        return self.cache[last_key]

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        raise NotImplementedError

    def _get_args_for_original_fn(self, args: Any) -> List[Any]:
        return args

    def _trace_fn(self, args):
        # returns a non-transformed trace
        inputs, inputs_pytree = tree_flatten(args)
        input_tensors = extract_input_tensors(inputs)
        outs, outs_pytree = tree_flatten(self.f(*self._get_args_for_original_fn(args)))

        assert len(input_tensors) == len(
            inputs
        ), "Input tensors should have the same length"
        inputs = input_tensors
        return GraphTrace(
            inputs=inputs,
            inputs_pytree=inputs_pytree,
            input_tensors=input_tensors,
            outputs=outs,
            outputs_pytree=outs_pytree,
        )

    def _get_maybe_cached_transformed_trace(self, args: List[Any]) -> GraphTrace:
        inputs, inputs_pytree = tree_flatten(args)
        input_tensors = extract_input_tensors(inputs)
        cache_key = get_cache_key(inputs, self.assume_static_argnums)
        if self.cache[cache_key] is not None:
            cached = self.cache[cache_key]
            inputs = [x for x in cached.inputs]
            cloned_output_tensors, cloned_input_tensors = clone_graph(
                extract_input_tensors(cached.outputs), cached.input_tensors
            )
            # make sure every element in inputs is substitutedb y corresponding element in cloned_input_tensors
            cloned_outputs = [x for x in cached.outputs]
            x = 0
            for i, out in enumerate(cached.outputs):
                if isinstance(out, Tensor):
                    cloned_outputs[i] = cloned_output_tensors[x]
                    x += 1
            x = 0
            for i, inp in enumerate(inputs):
                if isinstance(inp, Tensor):
                    inputs[i] = cloned_input_tensors[x]
                    x += 1
            return GraphTrace(
                inputs=inputs,
                inputs_pytree=cached.inputs_pytree,
                input_tensors=cloned_input_tensors,
                outputs=cloned_outputs,
                outputs_pytree=cached.outputs_pytree,
            )
        # TODO -- MIGHT CAUSE BUGS
        new_args = self._get_args_for_original_fn(args)
        outs, outs_pytree = tree_flatten(self.f(*new_args))
        # inputs, inputs_pytree = tree_flatten(new_args)
        # input_tensors = extract_input_tensors(inputs) # breaks vmap

        out_tensors = extract_input_tensors(outs)
        out_tensors, input_tensors = clone_graph(out_tensors, input_tensors)

        # using outs, create new outs that matches the shape of the original outs
        outs = [x for x in outs]
        x = 0
        for i, out in enumerate(outs):
            if isinstance(out, Tensor):
                outs[i] = out_tensors[x]
                x += 1

        inputs = [x for x in inputs]
        x = 0
        for i, inp in enumerate(inputs):
            if isinstance(inp, Tensor):
                inputs[i] = input_tensors[x]
                x += 1

        orig_trace = GraphTrace(
            inputs=inputs,
            inputs_pytree=inputs_pytree,
            input_tensors=input_tensors,
            outputs=outs,
            outputs_pytree=outs_pytree,
        )
        transformed_trace = self._transform_trace(orig_trace)
        self.cache[cache_key] = transformed_trace
        return self._get_maybe_cached_transformed_trace(args)

    def post_process_outs(self, outs, args, input_tensors):
        return outs

    def __call__(self, *args):
        orig_args = args
        trace = self._get_maybe_cached_transformed_trace(args)
        args, _ = tree_flatten(args)
        args = [x for x in args if isinstance(x, Tensor)]
        bridge_args_to_lazy_fn(trace.input_tensors, args)
        trace.outputs = self.post_process_outs(
            trace.outputs, orig_args, trace.input_tensors
        )
        ret = tree_unflatten(trace.outputs_pytree, trace.outputs)
        return ret

    def print_trace(self):
        # get last key in cache
        assert len(self.cache) > 0, "No cache entries"
        last_key = list(self.cache.keys())[-1]
        trace = self.cache[last_key]

        print_trace(trace)

    def get_constants(self):
        assert len(self.cache) > 0, "No cache entries"
        last_key = list(self.cache.keys())[-1]
        trace = self.cache[last_key]
        # constants are elements in the graph with primitive FromNumpy or ADPrimitive
        constants = []
        visited = set()

        def recurse(x):
            if x.id not in visited:
                for child in x.children():
                    recurse(child)
                if x.op == "FromNumpy" or x.op == "ADPrimitive":
                    constants.append(x)
                visited.add(x.id)

        for out in trace.outputs:
            recurse(out)
        return constants


class Identity(LazyFunction):
    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        return trace


# utils

# recurses the graph and executes a lambda passed as argument


def topo_recurse(x, fn):
    visited = set()

    def recurse(x):
        if x.id not in visited:
            for child in x.children():
                recurse(child)
            fn(x)
            visited.add(x.id)

    if isinstance(x, list):
        for xx in x:
            recurse(xx)
    else:
        recurse(x)


def topo_recurse_until_reach_inputs(x, fn, inputs, do_for_input=True):
    visited = set()
    inputs = [input.id for input in inputs]

    def recurse(x):
        if x.id in inputs:
            if do_for_input:
                fn(x)
            return
        if x.id not in visited:
            for child in x.children():
                recurse(child)
            fn(x)
            visited.add(x.id)

    if isinstance(x, (list, tuple)):
        for xx in x:
            recurse(xx)
    else:
        recurse(x)


import random


def get_random_possible_toposorts(x, inputs):
    visited = set()
    toposorted_tensors = []
    inputs = [input.id for input in inputs]

    def recurse(x):
        if x.id in inputs:
            toposorted_tensors.append(x)
            return
        if x.id not in visited:
            for child in random.sample(x.children(), len(x.children())):
                recurse(child)
            toposorted_tensors.append(x)
            visited.add(x.id)

    if isinstance(x, (list, tuple)):
        for xx in x:
            recurse(xx)
    else:
        recurse(x)
    return toposorted_tensors


def get_consumers(xs):
    consumers = {}

    def add_consumer(x):
        for child in x.children():
            if child.id not in consumers:
                consumers[child.id] = []
            consumers[child.id].append(x)

    topo_recurse(xs, add_consumer)
    return consumers


def deepcopy_graphtrace(trace):
    # uses the clone_graph function to clone the graph
    outputs, input_tensors = clone_graph(trace.outputs, trace.input_tensors)
    assert len(trace.inputs) == len(
        input_tensors
    ), "Input tensors should have the same length"
    return GraphTrace(
        inputs=input_tensors,
        inputs_pytree=trace.inputs_pytree,
        input_tensors=input_tensors,
        outputs=outputs,
        outputs_pytree=trace.outputs_pytree,
    )
