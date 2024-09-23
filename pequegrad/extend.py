import numpy as np
from pequegrad.backend.c import Tensor, dt
from typing import Tuple, Union, List
from pequegrad.backend.c import custom_prim as _custom_prim

_ArrayLike = Union[float, int, np.ndarray, "Tensor", List["_ArrayLike"]]
_Shape = Union[int, Tuple[int, ...]]
dtypetonp = {dt.float32: np.float32, dt.float64: np.float64, dt.int32: np.int32}


"""
Cpp code for primitive
class ADPrimitive {
public:
  virtual bool eager() { return false; }
  /**
   * Dispatch is responsible for allocating the memory, and populating
   * the `View` objects in the output tensors. The `View` objects have
   * the data ptrs, as well as strides/shape information and dtypes.
   */
  virtual void dispatch_cpu(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs);
  virtual void dispatch_cuda(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs);

  /**
   * The backward pass of the primitive. It does not really take care of the
   * computation, but uses other primitives to compute the gradients.
   * This allows for higher order derivatives.
   */
  virtual std::vector<Tensor> backward(const std::vector<Tensor> &primals,
                                       const std::vector<Tensor> &tangents,
                                       const std::vector<Tensor> &outputs);

  virtual std::vector<View> precompute(const std::vector<Tensor> &inputs) {
    throw std::runtime_error("precompute not implemented for " + str());
  }

  virtual std::string str() { return "ADPrimitive"; }
};
"""


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
