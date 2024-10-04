from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from contextvars import ContextVar
from .lazyfn import GraphTrace, LazyFunction, extract_input_tensors as extract_tensors

inside_jit = ContextVar("inside_jit", default=False)


class jit(LazyFunction):
    def __init__(self, f, opts=None):
        super().__init__(f)
        self.opts = opts if opts is not None else {}

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
