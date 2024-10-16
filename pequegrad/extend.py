from pequegrad.backend.c import Tensor
from pequegrad.backend.c import custom_prim as _custom_prim


def custom_prim(f):
    def ff(*args):
        res = f(*args)
        if isinstance(res, tuple):
            return res
        if isinstance(res, Tensor):
            return (res,)
        else:
            raise ValueError("custom_prim must return a Tensor or a tuple of Tensors")

    p = _custom_prim(ff)
    p.vjp = lambda f: p.setvjp(f)
    return p


class Primitive:
    @staticmethod
    def dispatch(inputs):
        raise NotImplementedError

    @staticmethod
    def backward(primals, tangents, outputs):
        raise NotImplementedError

    @staticmethod
    def precompute(inputs):
        raise NotImplementedError

    @classmethod
    def str(cls):
        return "WrappedPyPrimitive<{}>".format(cls.__name__)

    @classmethod
    def eager(cls):
        return False

    @classmethod
    def apply(cls, *args):
        def _fn(*inputs):
            ret = cls.dispatch(inputs)
            print("ret", ret)
            return ret

        def _vjpf(primals, tangents, outputs):
            return cls.backward(primals, tangents, outputs)

        prim = custom_prim(_fn)
        prim.setvjp(_vjpf)
        return prim(*args)
