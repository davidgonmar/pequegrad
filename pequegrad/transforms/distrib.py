from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from .lazyfn import GraphTrace, LazyFunction, deepcopy_graphtrace
from pequegrad.backend.c import grads  # noqa
from .pytree import (
    tree_flatten,
    PyTreeDef,
)  # noqa
from typing import List


def flatten_argnums(inputs_pytree: PyTreeDef, argnums: List[int]) -> List[int]:
    # flatten the argnums to the flattened structure of the inputs_pytree
    assert len(argnums) == 1, "Only one argnum supported"
    argnum = argnums[0]
    # inputs_pytree = inputs_pytree.structure
    flat, _ = tree_flatten(inputs_pytree.structure[argnum])
    flattened_start_index = len(tree_flatten(inputs_pytree.structure[:argnum])[0])
    flattened_indices = list(
        range(flattened_start_index, flattened_start_index + len(flat))
    )
    return flattened_indices

class pmap(LazyFunction):
    def __init__(self, f, devices, argnum_opts):
        super().__init__(f)
        self.devices = devices
        self.argnum_opts = argnum_opts

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # for each argnum, it can either be an int (shard) or None (no shard)
        inputs = trace.input_tensors
        flattened_argnums_opts = flatten_argnums(trace.inputs_pytree, self.argnum_opts)
        argnum = 0
        traces_per_device = {}
        # the process is:
        # first, create subgraphs for each device
        graph_per_devices = dict()
        for device in self.devices:
            graph_per_devices[device] = deepcopy_graphtrace(trace)

        # for sharded argnums, we need to split the input tensors
        # for the others, we replicate the input tensors
        apply_funcs = dict()
        for i, opt in enumerate(self.argnum_opts):
            if opt is not None:
                def apply_func(input):
                    split = input.split(len(self.devices), dim=0)
                    return [split.to(device) for device in self.devices]
            else:
                def apply_func(input):
                    return [input.to(device) for device in self.devices]

            apply_funcs[i] = apply_func
        
        # now, for each graphtrace, apply the apply_func for the input tensors.
        # also, we need to re-propagate shape/device information
        for device in self.devices:
            graph = graph_per_devices[device]
            for i, apply_func in apply_funcs.items():
                graph.input_tensors[i] = apply_func(graph.input_tensors[i])
                graph.inputs[i] = graph.input_tensors[i]
            traces_per_device[device] = graph

        # reprogate the shape/device information
        raise NotImplementedError
    
        # merge the traces
        raise NotImplementedError