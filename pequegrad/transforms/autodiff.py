from pequegrad.backend.c import grads
from pequegrad.backend.c import custom_prim as _custom_prim, Tensor
from .pytree import (
    make_pytree_nested_list,
    tree_flatten,
    PyTreeDef,
    make_pytree_list,
    pytree_def_to_dict,
)
from .lazyfn import LazyFunction, GraphTrace, topo_recurse_until_reach_inputs
import itertools
from typing import List, Callable


def ndindex(shape):
    if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape):
        raise ValueError("Shape must be a tuple of integers")
    return tuple(itertools.product(*(range(dim) for dim in shape)))


def standard_basis_and_ndindex(shape: List[int], device, dtype):
    idxs = ndindex(tuple(shape))
    return [
        Tensor.zeros(shape, device=device)
        .astype(dtype)
        .at[idx]
        .set(Tensor.ones([], device=device).astype(dtype))
        for idx in idxs
    ], idxs


def jacrev(out, wrt):
    # we can compute the jacobian by computing the gradient of each element of the output
    # that can be done by computing a vjp with v = e_i where e_i is the i-th element of the standard basis of the output space
    jacs = []
    if isinstance(wrt, Tensor):
        wrt = [wrt]
    basis, ndixs = standard_basis_and_ndindex(out.shape, out.device, out.dtype)
    for w in wrt:
        jac = Tensor.zeros((*out.shape, *w.shape), device=out.device)
        for elem, ndidx in zip(basis, ndixs):
            g = grads([w], out, elem)[0]
            jac = jac.at[ndidx].set(g)
        jacs.append(jac)
    return jacs


def hessian(out, wrt):
    # hessian can be seen as the jacobian of the jacobian
    jacs = jacrev(out, wrt)
    hesss = []
    for jac in jacs:
        hesss.append(jacrev(jac, wrt))
    return hesss


def flatten_argnums(inputs_pytree: PyTreeDef, argnums: List[int]) -> List[int]:
    # flatten the argnums to the flattened structure of the inputs_pytree
    assert len(argnums) == 1, "Only one argnum supported"
    argnum = argnums[0]
    # inputs_pytree = inputs_pytree.structure
    flat, _ = tree_flatten(pytree_def_to_dict(inputs_pytree.structure[argnum]))
    rest = pytree_def_to_dict(inputs_pytree.structure[:argnum])
    if len(rest) > 0:
        flattened_start_index = len(tree_flatten(rest)[0])
    else:
        flattened_start_index = 0
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
        # returned grads have shape of the wrt (maybe a dict for named parameters)
        # so we get the inputs_pytree of the wrt
        wrt_pytree = make_pytree_list(
            [trace.inputs_pytree.structure[i] for i in self.wrt]
        )
        if self.return_outs:
            new_outs_pytree = PyTreeDef(
                type=tuple, structure=[trace.outputs_pytree, wrt_pytree]
            )
        else:
            new_outs_pytree = wrt_pytree
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
        super().__init__(f)
        self.wrt = wrt
        self.return_outs = return_outs

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # fnhessian returns the same trace, but the outputs -> outputs, hessian
        fn_out = trace.outputs
        assert len(fn_out) == 1, "Only one output supported"
        flattened_indices = []
        for xxx in self.wrt:
            flattened_indices.extend(flatten_argnums(trace.inputs_pytree, [xxx]))
        wrt = [trace.inputs[i] for i in flattened_indices]
        hess = hessian(fn_out[0], wrt)
        flattened_hess = []
        for h in hess:
            for hh in h:
                flattened_hess.append(hh)
        new_outs = fn_out + flattened_hess if self.return_outs else flattened_hess
        new_outs_pytree = None
        if self.return_outs:
            new_outs_pytree = PyTreeDef(
                type=tuple,
                structure=[
                    trace.outputs_pytree,
                    make_pytree_nested_list(len(wrt), len(wrt)),
                ],
            )
        else:
            # if we have 3 argnums, the result is a list of lists of tensors of size (3, 3)
            new_outs_pytree = PyTreeDef(
                type=list,
                structure=[
                    PyTreeDef(
                        type=list,
                        structure=[make_pytree_nested_list(len(wrt), len(wrt))],
                    )
                ],
            )
        return GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=new_outs,
            outputs_pytree=new_outs_pytree,
        )


_vjp = grads


def _vhp(out, wrt, v):
    # vhp = hessian(out, wrt) @ v
    # assuming out is a scalar, vhp = vjp(vjp(out, wrt), v)
    # to match the torch behaviour, we accumulate the second derivatives
    assert len(out.shape) == 0, "Output must be a scalar"
    res = []

    def _acc(x, y):
        return x + y

    for arg in wrt:
        vjp = _vjp([arg], out)[0]
        curres = Tensor.zeros(arg.shape, device=arg.device)
        for arg2 in wrt:
            vjp2 = _vjp([arg2], vjp, v)[0]
            curres = _acc(curres, vjp2)
        res.append(curres)  # as seen, we accumulate over the second derivatives
    return res


# Vector-Hessian product
class fnvhp(LazyFunction):
    def __init__(self, f, wrt, return_outs=False):
        super().__init__(f)
        self.wrt = wrt
        self.return_outs = return_outs

    def _get_args_for_original_fn(self, args):
        # all args except last one
        return args[:-1]

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # fnvhp returns the same trace, but the outputs -> outputs, vhp
        fn_out = trace.outputs

        """
        def f(a, b):
            return (a * b).sum()

        hessianfn = fnhessian(f, wrt=[0, 1], return_outs=True)

        # transformed fn
        def f(a, b, v): # notice the extra v
            return (a * b).sum(), some_computation(a, b, v)

        """
        assert len(fn_out) == 1, "Only one output supported"
        flattened_indices = []
        for xxx in self.wrt:
            flattened_indices.extend(flatten_argnums(trace.inputs_pytree, [xxx]))
        wrt = [trace.inputs[i] for i in flattened_indices]
        v = trace.inputs[-1]
        vhp = _vhp(fn_out[0], wrt, v)
        new_outs = fn_out + vhp if self.return_outs else vhp
        new_outs_pytree = None
        if self.return_outs:
            new_outs_pytree = PyTreeDef(
                type=tuple,
                structure=[
                    trace.outputs_pytree,
                    make_pytree_list(wrt),
                ],
            )
        else:
            new_outs_pytree = PyTreeDef(
                type=list,
                structure=[
                    PyTreeDef(
                        type=list,
                        structure=[make_pytree_list(wrt)],
                    )
                ],
            )
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


# ===================== NEW CHECKPOINT THING =====================
def checkpoint(f: Callable, diff_argnums=None) -> Callable:
    if diff_argnums is None:
        import inspect

        diff_argnums = list(range(len(inspect.signature(f).parameters)))
    prim = custom_prim(f)
    grad_fn = fngrad(f, wrt=diff_argnums)

    def grad_fn_for_setvjp(primals, tangents, outputs):
        ret = grad_fn(*primals)
        return ret

    prim.setvjp(grad_fn_for_setvjp)

    return prim


# ===================== FORWARD AUTODIFF =====================


def add_jvp(primals, tangents, argnums, outputs):
    return tangents[0]


def sub_jvp(primals, tangents, argnums, outputs):
    return tangents[0]


def broadcast_jvp(primals, tangents, argnums, outputs):
    if outputs[0].shape == primals[0].shape:
        return tangents[0]
    else:
        raise ValueError("Broadcasting not supported for JVP")


def mul_jvp(primals, tangents, argnums, outputs):
    if len(argnums) == 1:
        return primals[1 - argnums[0]] * tangents[0]
    else:
        return primals[1] * tangents[0] + primals[0] * tangents[1]


jvp_rules = {"Add": add_jvp, "Sub": sub_jvp, "Broadcast": broadcast_jvp, "Mul": mul_jvp}


def get_jvp_graph(trace: GraphTrace):
    # basically gets a graph and returns a new graph with the jvp only

    tangents = dict()
    for inp in trace.inputs:
        tangents[inp.id] = Tensor.ones(inp.shape, device=inp.device)

    toposorted = []  # input tensors are first

    def _fn(tensor):
        toposorted.append(tensor)

    topo_recurse_until_reach_inputs(trace.outputs[0], _fn, trace.inputs)

    for tensor in toposorted:  # input tensors are first, until outputs
        if tensor.id in tangents:
            continue
        if tensor.ad_context() in jvp_rules:
            primals = tensor.children()
            _tangents = []
            _argnums = []
            for idx, p in enumerate(primals):
                assert p.id in tangents, f"Primal {p.id} not in tangents: {tangents}"
                _tangents.append(tangents[p.id])
                _argnums.append(idx)
            tan_ = jvp_rules[tensor.ad_context()](
                primals, _tangents, _argnums, [tensor]
            )
            tangents[tensor.id] = tan_
        else:
            raise ValueError(f"Operation {tensor.ad_context()} not supported for JVP")

    return tangents[trace.outputs[0].id]


class jvp(LazyFunction):
    def __init__(self, f):
        super().__init__(f)

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        fn_out = trace.outputs
        assert len(fn_out) == 1, "Only one output supported"
        jvp = get_jvp_graph(trace)
        new_outs = fn_out + [jvp]
        new_outs_pytree = PyTreeDef(
            type=tuple,
            structure=[trace.outputs_pytree, trace.outputs_pytree],
        )
        ret = GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=new_outs,
            outputs_pytree=new_outs_pytree,
        )
        return ret


def value_and_grad(f, wrt=[0]):
    return fngrad(f, wrt, return_outs=True)


class vjp(LazyFunction):
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
        cotangent = trace.inputs[-1]
        grad = grads(wrt, fn_out[0], cotangent)
        assert len(grad) == len(wrt), "Gradient and wrt must have the same length"
        new_outs = fn_out + grad if self.return_outs else grad
        new_outs_pytree = None

        # returned grads have shape of the wrt (maybe a dict for named parameters)
        # so we get the inputs_pytree of the wrt
        wrt_pytree = make_pytree_list(
            [trace.inputs_pytree.structure[i] for i in self.wrt]
        )
        if self.return_outs:
            new_outs_pytree = PyTreeDef(
                type=tuple, structure=[trace.outputs_pytree, wrt_pytree]
            )
        else:
            new_outs_pytree = wrt_pytree
        return GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=new_outs,
            outputs_pytree=new_outs_pytree,
        )

    def _get_args_for_original_fn(self, args):
        # all args except last one
        return args[:-1]


def jvp(f, wrt):
    assert len(wrt) == 1, "Only one wrt supported"

    def new_function(*args):
        v = args[-1]
        args = args[:-1]

        def _f(*args):
            # v = args[-1]
            vjp_res = vjp(f, wrt=wrt)(*args)
            return vjp_res[0]  # returns (u * Jf)

        vjp2 = vjp(_f, wrt=[len(args)])(*args, v, v)  # returns (Jf) * v
        return vjp2

    return new_function
