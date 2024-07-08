from pequegrad.autodiff import *  # noqa
from pequegrad.tensor import *  # noqa
from pequegrad.optim import *  # noqa
from pequegrad.modules import *  # noqa
from pequegrad.einsum import *  # noqa
from pequegrad.compile import *  # noqa
from pequegrad.backend.c import dt, device, custom_prim as _custom_prim  # noqa
from pequegrad.ops import *  # noqa


def custom_prim(f, compile_jit=False):
    def ff(*args):
        res = f(*args)
        if isinstance(res, tuple):
            return res
        if isinstance(res, Tensor):
            return (res,)
        else:
            raise ValueError("custom_prim must return a Tensor or a tuple of Tensors")

    if compile_jit:
        ff = jit(ff)

    p = _custom_prim(ff)

    # not supported yet
    if compile_jit:
        p.vjp = lambda f: p.setvjp(jit(f))
    else:
        p.vjp =  lambda f: p.setvjp(f)
    
    return p
