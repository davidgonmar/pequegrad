from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from contextvars import ContextVar
from .lazyfn import GraphTrace, LazyFunction, topo_recurse, get_consumers
import pequegrad.ops as ops

inside_jit = ContextVar("inside_jit", default=False)



_ops_cast_to_fp16 = {
    "MatMul"
}
_str_to_op = {
    "MatMul": ops.matmul
}

def get_new_op(tensor):
    if tensor.ad_context() in _ops_cast_to_fp16 and tensor.dtype == dt.float32:
        tchildren = tensor.children()
        casted = [t.astype("float16") for t in tchildren]  

        new_op = _str_to_op[tensor.primitive().str()]

        return new_op(*casted).astype("float32")    
    return None

class amp(LazyFunction):
    def __init__(self, f, opts=None):
        super().__init__(f)
        self.opts = opts if opts is not None else {}

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # same as autograd, but it just compiles the graph

        consumers = get_consumers(trace.outputs)
        def recurse_fn(tensor):
            new_t = get_new_op(tensor)
            if new_t is not None:
                # replace the tensor in the consumers
                for consumer in consumers.get(tensor, []):
                    consumer.replace_child(tensor, new_t)
                # if it is an output, replace it in the outputs
                
                if (any(tensor.id ==  t.id for t in trace.outputs)):
                    trace.outputs[trace.outputs.index(tensor)] = new_t
        topo_recurse(trace.outputs, recurse_fn)
        new_trace = GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=trace.outputs,
            outputs_pytree=trace.outputs_pytree,
        )
        return new_trace
