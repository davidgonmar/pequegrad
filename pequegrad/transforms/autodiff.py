from pequegrad.backend.c import grads  # noqa
from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
from .pytree import (
    tree_flatten,
    tree_unflatten,
    PyTreeDef,
    make_pytree_list,
    is_module,
)  # noqa
from .lazyfn import LazyFunction, GraphTrace, Cache  # noqa
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
        self.f = f
        self.return_outs = return_outs
        self.wrt = wrt

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # fngrad returns the same trace, but the outputs -> outputs, grads if return_outs is True, else grads
        fn_out = trace.outputs
        assert len(fn_out) == 1, "Only one output supported"
        flattened_indices = flatten_argnums(trace.inputs_pytree, self.wrt)
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
            new_outs_pytree = trace.outputs_pytree

        return GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=new_outs,
            outputs_pytree=new_outs_pytree,
        )


class fnjacobian(LazyFunction):
    def __init__(self, f, wrt, return_outs=False):
        self.f = f
        self.wrt = wrt
        self.return_outs = return_outs

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # fnjacobian returns the same trace, but the outputs -> outputs, jacobian
        fn_out = trace.outputs
        assert len(fn_out) == 1, "Only one output supported"
        flattened_indices = flatten_argnums(trace.inputs_pytree, self.wrt)
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
        print(new_outs_pytree)
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
