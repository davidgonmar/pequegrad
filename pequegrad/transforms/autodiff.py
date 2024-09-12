from pequegrad.backend.c import grads  # noqa
from pequegrad.backend.c import (
    Tensor,
    custom_prim as _custom_prim,
)  # noqa
from .pytree import (
    tree_flatten,
    PyTreeDef,
    make_pytree_list,
)  # noqa
from .lazyfn import (
    LazyFunction,
    GraphTrace,
    Cache,
    get_cache_key,
)  # noqa
import itertools
from typing import List


def ndindex(shape):
    if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError("Shape must be a tuple of integers")
    return itertools.product(*(range(dim) for dim in shape))


def jacrev(out, wrt):
    # we can compute the jacobian by computing the gradient of each element of the output
    # that can be done by computing a vjp with v = e_i where e_i is the i-th unit vector
    jacs = []
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

    return jacs


def hessian(out, wrt):
    # hessian can be seen as the jacobian of the gradient!
    gs = grads(wrt, out)[0]
    return jacrev(gs, wrt)


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


class fngrad(LazyFunction):
    def __init__(self, f, wrt, return_outs=False):
        super().__init__(f)
        self.wrt = wrt
        self.return_outs = return_outs

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # fngrad returns the same trace, but the outputs -> outputs, grads if return_outs is True, else grads
        fn_out = trace.outputs
        assert len(fn_out) == 1, "Only one output supported"
        flattened_indices = []
        for xxx in self.wrt:
            flattened_indices.extend(flatten_argnums(trace.inputs_pytree, [xxx]))
        wrt = [trace.inputs[i] for i in flattened_indices]
        grad = grads(wrt, fn_out[0])
        assert len(grad) == len(wrt), "Gradient and wrt must have the same length"
        new_outs = fn_out + grad if self.return_outs else grad
        new_outs_pytree = None
        if self.return_outs:
            new_outs_pytree = PyTreeDef(
                type=tuple, structure=[trace.outputs_pytree, make_pytree_list(wrt)]
            )
        else:
            new_outs_pytree = PyTreeDef(type=list, structure=[make_pytree_list(wrt)])

        return GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=new_outs,
            outputs_pytree=new_outs_pytree,
        )


class fnjacobian(LazyFunction):
    def __init__(self, f, wrt, return_outs=False):
        super().__init__(f)
        self.wrt = wrt
        self.return_outs = return_outs

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # fnjacobian returns the same trace, but the outputs -> outputs, jacobian
        fn_out = trace.outputs
        assert len(fn_out) == 1, "Only one output supported"
        flattened_indices = []
        for xxx in self.wrt:
            flattened_indices.extend(flatten_argnums(trace.inputs_pytree, [xxx]))
        wrt = [trace.inputs[i] for i in flattened_indices]
        jac = jacrev(fn_out[0], wrt)
        new_outs = fn_out + jac if self.return_outs else jac
        new_outs_pytree = None
        if self.return_outs:
            new_outs_pytree = PyTreeDef(
                type=tuple, structure=[trace.outputs_pytree, make_pytree_list(wrt)]
            )
        else:
            new_outs_pytree = trace.outputs_pytree
        return GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=new_outs,
            outputs_pytree=new_outs_pytree,
        )


class fnhessian(LazyFunction):
    def __init__(self, f, wrt, return_outs=False):
        self.f = f
        self.wrt = wrt
        self.return_outs = return_outs

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # fnhessian returns the same trace, but the outputs -> outputs, hessian
        fn_out = trace.outputs
        assert len(fn_out) == 1, "Only one output supported"
        flattened_indices = flatten_argnums(trace.inputs_pytree, self.wrt)
        wrt = [trace.inputs[i] for i in flattened_indices]
        hess = hessian(fn_out[0], wrt)
        new_outs = fn_out + hess if self.return_outs else hess
        new_outs_pytree = None
        if self.return_outs:
            new_outs_pytree = PyTreeDef(
                type=tuple, structure=[trace.outputs_pytree, make_pytree_list(wrt)]
            )
        else:
            new_outs_pytree = trace.outputs_pytree
        return GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=new_outs,
            outputs_pytree=new_outs_pytree,
        )


# Given a graph, to compute its gradients we usually keep the graphs' intermediate values in memory.
# Gradient checkpointing can be seen as 'duplicating' the graph, then computing the gradients of the duplicated graph.

"""
def f(a, b, c):
    x = a * b
    return x * c

g = fngrad(f, wrt=[0, 1, 2])

def g(a, b, c):
    # original graph
    x = a * b
    ret = x * c

    # gradient graph
    grad_ret_wrt_x = c
    grad_ret_wrt_c = x
    grad_ret_wrt_a = grad_ret_wrt_x * b
    grad_ret_wrt_b = grad_ret_wrt_x * a

    grads = [grad_ret_wrt_a, grad_ret_wrt_b, grad_ret_wrt_c]

    # as seen, we need to keep x in memory even after we have computed ret
    return ret, grads

def g_after_checkpoint(a, b, c):
    # original graph
    x = a * b
    ret = x * c

    x_fake = a * b
    ret_fake = x_fake * c

    # gradient graph
    grad_ret_wrt_x = c
    grad_ret_wrt_c = x_fake
    grad_ret_wrt_a = grad_ret_wrt_x * b
    grad_ret_wrt_b = grad_ret_wrt_x * a

    grads = [grad_ret_wrt_a, grad_ret_wrt_b, grad_ret_wrt_c]

    # here, grad_ret_wrt_c does not have a dependency on x
    # this is achieved by recomputing x in the gradient graph

    return ret, grads
"""


def custom_prim(f):
    def ff(*args):
        res = f(*args)
        if isinstance(res, (tuple, list)):
            return res
        if isinstance(res, Tensor):
            return (res,)
        else:
            raise ValueError(
                "custom_prim must return a Tensor or a tuple of Tensors, got {}".format(
                    res
                )
            )

    p = _custom_prim(ff)

    p.vjp = lambda x: p.setvjp(x)
    return p


# does nothing at all
class tracedfn(LazyFunction):
    def __init__(self, f):
        self.f = f

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        return trace

    # functionalizes a trace
    @staticmethod
    def from_trace(trace: GraphTrace):
        self = tracedfn(None)

        def _f(*args):
            raise ValueError("Traced function cannot be called")

        self.cache = Cache()
        self.cache[get_cache_key(trace.inputs)] = trace
        self.f = _f
        return self


class checkpoint(LazyFunction):
    def __init__(self, f):
        self.f = f

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # orig_traced_f is a function representation of the trace
        orig_traced_f = tracedfn.from_trace(trace)
        orig_traced_f(*trace.input_tensors)
        # orig_traced_f.print_trace()
        """orig_f_grads = fngrad(orig_traced_f, wrt=[0])
        orig_f_grads(trace.input_tensors)
        
        orig_f_grads.print_trace()
        """
        # now, the original grad graph is in orig_f_grads cache
        # grads_trace = orig_f_grads.get_last_trace()

        # orig_f_grads.print_trace()
        wrt = list(range(len(trace.inputs)))
        grad_fn = fngrad(orig_traced_f, wrt=wrt)
        # INSPECT THE GRADIENT FUNCTION
        grad_fn(*trace.input_tensors)  # so it records the trace
        grad_fn = tracedfn.from_trace(
            grad_fn.get_last_trace()
        )  # for some reason, we need to do this?
        """std::vector<Tensor> ADPrimitive::backward(const std::vector<Tensor> &primals,
                                          const std::vector<Tensor> &tangents,
                                          const std::vector<Tensor> &outputs) {
        throw std::runtime_error("backward not implemented for " + str());
        }"""

        def grad_fn_for_setvjp(primals, tangents, outputs):
            ret = grad_fn(*primals)
            return ret[0]

        fff = custom_prim(tracedfn.from_trace(trace))
        fff.setvjp(grad_fn_for_setvjp)

        fff_out = fff(*trace.input_tensors)

        new_trace_inputs = trace.inputs
        new_trace_inputs_pytree = trace.inputs_pytree
        new_trace_input_tensors = trace.input_tensors
        new_trace_outputs = (
            fff_out if isinstance(fff_out, (tuple, list)) else (fff_out,)
        )
        new_trace_outputs_pytree = trace.outputs_pytree

        # this is a hack
        # now, look for the fff node in the graph, and replace it with a copy of the original trace
        ret = GraphTrace(
            inputs=new_trace_inputs,
            inputs_pytree=new_trace_inputs_pytree,
            input_tensors=new_trace_input_tensors,
            outputs=new_trace_outputs,
            outputs_pytree=new_trace_outputs_pytree,
        )

        # raise ValueError("Checkpoint not implemented yet")

        # print_trace(ret)
        return ret
