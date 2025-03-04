import jax
from jax.tree_util import tree_flatten
from jax._src.util import safe_map
from pequegrad import ops
import pequegrad as pg


class Env:
    def __init__(self):
        self.env = {}

    def __getitem__(self, key):
        key = str(key)
        return self.env[key]

    def __setitem__(self, key, value):
        key = str(key)
        self.env[key] = value

    def __contains__(self, key):
        key = str(key)
        return key in self.env

    def get(self, key, default=None):
        key = str(key)
        return self.env.get(key, default)


@pg.jit
def pg_tensor_interp(jaxpr, device, consts, *args):
    env = Env()
    for var, const in zip(jaxpr.constvars, consts):
        env[var] = const
    flat_args, _ = tree_flatten(args)
    for var, arg in zip(jaxpr.invars, flat_args):
        env[var] = arg
    for eqn in jaxpr.eqns:
        in_vals = [
            v.val if isinstance(v, jax.core.Literal) else env[v] for v in eqn.invars
        ]
        in_vals = [
            (
                v
                if isinstance(v, pg.Tensor)
                else pg.fill((), pg.dt.float32, v.item(), device)
            )
            for v in in_vals
        ]
        prim_name = eqn.primitive.name
        if prim_name == "dot_general":
            dimension_numbers = eqn.params.get(
                "dimension_numbers", (([in_vals[0].ndim - 1], [0]), ((), ()))
            )
            (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
            if len(lhs_batch) == 0 and len(rhs_batch) == 0:
                res = ops.tensordot(
                    in_vals[0], in_vals[1], dims=(lhs_contract, rhs_contract)
                )
            else:
                raise NotImplementedError(
                    "dot_general with batch dimensions is not implemented."
                )
        elif prim_name == "add":
            res = ops.add(in_vals[0], in_vals[1])
        elif prim_name == "tanh":
            res = ops.tanh(in_vals[0])
        elif prim_name == "mul":
            res = ops.mul(in_vals[0], in_vals[1])
        elif prim_name == "sub":
            res = ops.sub(in_vals[0], in_vals[1])
        elif prim_name == "power":
            exponent = eqn.params.get("exponent", 2)
            res = ops.pow(in_vals[0], exponent)
        elif prim_name == "mean":
            res = ops.mean(in_vals[0], axes=eqn.params.get("axis"))
        elif prim_name == "neg":
            res = -in_vals[0]
        elif prim_name == "div":
            res = in_vals[0] / in_vals[1]
        elif prim_name == "reshape":
            newshape = eqn.params.get("new_sizes")
            res = in_vals[0].reshape(newshape)
        elif prim_name == "broadcast_in_dim":
            shape = eqn.params.get("shape")
            res = ops.broadcast_to(in_vals[0], shape)
        elif prim_name == "reduce_sum":
            axes = eqn.params.get("axes", None)
            keepdims = eqn.params.get("keepdims", False)
            res = ops.sum(in_vals[0], axes=axes, keepdims=keepdims)
        elif prim_name == "transpose":
            permutation = eqn.params.get("permutation")
            res = ops.permute(in_vals[0], permutation)
        elif prim_name == "convert_element_type":
            res = in_vals[0]
        elif prim_name == "integer_pow":
            res = ops.pow(in_vals[0], eqn.params["y"])
        elif prim_name == "add_any":
            result = in_vals[0]
            for operand in in_vals[1:]:
                result = ops.add(result, operand)
            res = result
        elif prim_name == "copy":
            res = in_vals[0].copy()
        else:
            raise NotImplementedError(
                f"Primitive '{prim_name}' not implemented in numpy_interp."
            )
        if len(eqn.outvars) == 1:
            env[eqn.outvars[0]] = res
        else:
            for var, r in zip(eqn.outvars, res):
                env[var] = r

    return safe_map(env.get, jaxpr.outvars)


def eval_jaxpr(jaxpr, consts, *args, device):
    env = Env()
    # Bind constants and inputs.
    for var, const in zip(jaxpr.constvars, consts):
        env[var] = pg.Tensor(const, device=device).astype("float32")
    flat_args, _ = tree_flatten(args)
    for var, arg in zip(jaxpr.invars, flat_args):
        env[var] = pg.Tensor(arg, device=device).astype(
            "float32"
        )  # TODO -- some way of extracting the type from the jaxpr
    _args = [env[var] for var in jaxpr.invars]
    _consts = [env[var] for var in jaxpr.constvars]
    return pg_tensor_interp(jaxpr, device, _consts, *_args)


def jax_jit(device="cuda"):
    if isinstance(device, str):
        device = pg.device.from_str(device)

    def inner(fun):
        # get in and out pytrees
        in_pytree = None
        out_pytree = None

        def wrapper(*args):
            nonlocal in_pytree, out_pytree
            if in_pytree is None:
                in_pytree = jax.tree_structure(args)
            if out_pytree is None:
                out_pytree = jax.tree_structure(fun(*args))
            jaxpr_obj = jax.make_jaxpr(fun)(*args)
            res = eval_jaxpr(jaxpr_obj.jaxpr, jaxpr_obj.literals, *args, device=device)
            res_np = jax.tree_map(lambda x: x.numpy(), res)
            return jax.tree_unflatten(out_pytree, res_np)

        return wrapper

    return inner
