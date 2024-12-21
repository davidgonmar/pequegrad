from .lazyfn import (
    LazyFunction,
    GraphTrace,
    print_trace,
    topo_recurse_until_reach_inputs,
)
from typing import Any, List
import pequegrad.ops as ops


def _bin_rule(a, b, axes):
    return a + b, axes


binops = {"Add", "Mul"}


class vmap(LazyFunction):
    def __init__(self, f, axes):
        super().__init__(f)
        self.axes = axes
        self.last_arg_shapes = None

    def _get_rule(self, x):
        if x in binops:
            return _bin_rule
        if x == "Broadcast":
            return lambda a, axes: (a, axes)
        else:
            raise NotImplementedError(f"Operation {x} not supported")

    def _get_args_for_original_fn(self, args: Any) -> List[Any]:
        new_args = []
        self.last_arg_shapes = []
        for idx, tensor in enumerate(args):
            # remove the axes
            shape = list(tensor.shape)
            shape.pop(self.axes[idx])
            new_args.append(ops.fill(shape, tensor.dtype, 0, tensor.device))
            self.last_arg_shapes.append(list(tensor.shape))
        return new_args

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        fn_out = trace.outputs
        inps = trace.inputs
        assert len(fn_out) == 1, "Only one output supported"
        print_trace(trace)
        substitution_map = {}

        def _rec(tensor):
            rule = self._get_rule(tensor.op)
            tinputs = tensor.children()
            axes = []
            new_inputs = []
            for inp in tinputs:
                new_inputs.append(substitution_map[inp.id]["tensor"])
                axes.append(substitution_map[inp.id]["axes"])

            new_out, new_axes = rule(*new_inputs, axes)

            substitution_map[tensor.id] = {"tensor": new_out, "axes": new_axes}

        for idx, inp in enumerate(inps):
            substitution_map[inp.id] = {
                # insert a 1 in the axis
                "tensor": inp.reshape(
                    list(inp.shape[: self.axes[idx]])
                    + [self.last_arg_shapes[idx][self.axes[idx]]]
                    + list(inp.shape[self.axes[idx] :])
                ),
                "axes": self.axes[idx],
            }
        topo_recurse_until_reach_inputs(fn_out, _rec, inps, do_for_input=False)
        new_inputs = []
        for inp in inps:
            new_inputs.append(substitution_map[inp.id]["tensor"])
        new_outs = []
        for out in fn_out:
            new_outs.append(substitution_map[out.id]["tensor"])
        ret = GraphTrace(
            inputs=new_inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=new_inputs,
            outputs=new_outs,
            outputs_pytree=trace.outputs_pytree,
        )
        print_trace(ret)
        return ret
