from pequegrad.backend.c import (
    compile,
    clone_graph,
    Tensor,
    dt,
    tensor_precompute_again,
    BroadcastTo as BroadcastToPrimitive,
)  # noqa
from .lazyfn import (
    GraphTrace,
    LazyFunction,
    deepcopy_graphtrace,
    topo_recurse_until_reach_inputs,
    get_consumers,
)
from pequegrad.backend.c import grads  # noqa
from .pytree import (
    tree_flatten,
    PyTreeDef,
)  # noqa
from typing import List
from pequegrad.distrib import reduce_to_one_device


def flatten_argnums(inputs_pytree: PyTreeDef, argnums: List[int]) -> List[int]:
    # flatten the argnums to the flattened structure of the inputs_pytree
    flattened_indices = []
    for argnum in argnums:
        # inputs_pytree = inputs_pytree.structure
        flat, _ = tree_flatten(inputs_pytree.structure[argnum])
        flattened_start_index = len(tree_flatten(inputs_pytree.structure[:argnum])[0])
        flattened_indices.append(
            list(range(flattened_start_index, flattened_start_index + len(flat)))
        )
    return flattened_indices


class pmap(LazyFunction):
    def __init__(self, f, devices, argnum_opts):
        super().__init__(f)
        self.devices = devices
        self.argnum_opts = argnum_opts

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        assert len(set([str(d) for d in self.devices])) == len(
            self.devices
        ), "Devices must be unique"

        # for each argnum, it can either be an int (shard) or None (no shard)
        numdevices = len(self.devices)
        inputs = trace.input_tensors
        argnums_correspondences = flatten_argnums(
            trace.inputs_pytree, range(len(self.argnum_opts))
        )
        argnum_opts = [
            self.argnum_opts[i]
            for i in range(len(self.argnum_opts))
            for _ in range(len(argnums_correspondences[i]))
        ]
        traces_per_device = dict()
        # new inputs are splitted if needed (argnum_opts is not None) or replicated (argnum_opts is None)
        new_inputs = []
        assert len(argnum_opts) == len(trace.input_tensors)
        for i, opt in enumerate(argnum_opts):
            if opt is not None:

                def _split_to_dev(old_input):
                    assert (
                        old_input.shape[opt] == numdevices
                    ), f"Splitting dimension must be equal to the number of devices: {old_input.shape[opt]} != {numdevices}"
                    return list(
                        map(
                            lambda x: x[0].to(self.devices[x[1]]),
                            zip(old_input.split(1, dim=opt), range(numdevices)),
                        )
                    )

                new_inputs.append(_split_to_dev(inputs[i]))
            else:
                new_inputs.append([inputs[i].to(device) for device in self.devices])

        for device in self.devices:
            traces_per_device[device] = deepcopy_graphtrace(trace)

        for deviceidx, device in enumerate(self.devices):
            consumers = get_consumers(traces_per_device[device].outputs)
            for i in range(len(trace.input_tensors)):
                old_consumers = consumers.get(
                    traces_per_device[device].input_tensors[i], []
                )
                old_tensor = traces_per_device[device].input_tensors[i]
                graph = traces_per_device[device]
                graph.input_tensors[i] = new_inputs[i][deviceidx]
                graph.inputs[i] = graph.input_tensors[i]
                traces_per_device[device] = graph
                # for every consumer of the old input tensor, replace it with the new input tensor
                for consumer in old_consumers:
                    consumer.replace_child(old_tensor, new_inputs[i][deviceidx])

        # since we are replicating/resharding the inputs, we need to repropagate the shape/device information
        for device in self.devices:

            def _todo_safe_precompute(x):
                tensor_precompute_again(x)

            topo_recurse_until_reach_inputs(
                traces_per_device[device].outputs,
                _todo_safe_precompute,
                traces_per_device[device].input_tensors,
            )

        # now, we "connect" thje original trace -> new_inputs -> traces_per_device -> new_outputs -> all_reduce -> outputs
        # connecting the orignal trace, new_inputs, and traces_per_device is done already
        # now, we need to connect the outputs of the traces_per_device to the all_reduce
        outputs = []
        for outidx in range(len(trace.outputs)):
            tensors = [
                traces_per_device[device].outputs[outidx] for device in self.devices
            ]
            outputs.append(reduce_to_one_device(tensors, "avg"))

        final_trace = GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=outputs,
            outputs_pytree=trace.outputs_pytree,
        )

        return final_trace
