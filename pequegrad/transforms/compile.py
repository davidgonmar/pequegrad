from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from contextvars import ContextVar
from .lazyfn import GraphTrace, LazyFunction, extract_input_tensors as extract_tensors

inside_jit = ContextVar("inside_jit", default=False)


class jit(LazyFunction):
    def __init__(self, f, assume_static_argnums=None, eval_outs=True, opts=None):
        super().__init__(f, assume_static_argnums)
        self.opts = opts if opts is not None else {}
        self.eval_outs = eval_outs

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # same as autograd, but it just compiles the graph
        new_trace = GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=trace.outputs,
            outputs_pytree=trace.outputs_pytree,
        )
        compile(extract_tensors(new_trace.outputs), self.opts)

        return new_trace

    def post_process_outs(self, outs):
        if self.eval_outs:
            outs = [o.eval() if isinstance(o, Tensor) else o for o in outs]
        return outs
