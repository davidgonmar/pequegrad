import pequegrad.ops as ops
from pequegrad.tensor import Tensor
from functools import wraps


def downsample_extend(f):
    @wraps(f)
    def wrapper(input: Tensor, k: int, *args, **kwargs):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        img_shape = input.shape
        assert (
            len(img_shape) == 3 or len(img_shape) == 4
        ), "Input must be a 3D or 4D tensor"
        if len(img_shape) == 3:
            input = input.unsqueeze(0)
        # img has shape (B, C, H, W)
        assert (
            input.shape[2] % k == 0 and input.shape[3] % k == 0
        ), "Input shape must be divisible by k, got shape: {} and k: {}".format(
            input.shape, k
        )
        ret = f(input, k, *args, **kwargs)
        if len(img_shape) == 3:
            ret = ret.squeeze(0)
        return ret

    return wrapper


@downsample_extend
def downsample_nearest(input: Tensor, k: int):
    # I(x, y) = input[x * k, y * k]
    return input[:, :, ::k, ::k]


@downsample_extend
def downsample_avgpooling(input: Tensor, k: int):
    # I(x, y) = 1/k^2 * sum_{i=0}^{k-1} sum_{j=0}^{k-1} input[x * k + i, y * k + j]
    return ops.avg_pool2d(input, k, k)


@downsample_extend
def downsample_bilinear(input: Tensor, k: int):
    h, w = input.shape[2], input.shape[3]
    h_new, w_new = h // k, w // k

    x = ops.linspace(0, h - 1, h_new, device=input.device)
    y = ops.linspace(0, w - 1, w_new, device=input.device)

    x0, x1 = ops.floor(x).astype("int32"), ops.ceil(x).astype("int32")
    y0, y1 = ops.floor(y).astype("int32"), ops.ceil(y).astype("int32")

    x0, x1 = ops.clip(x0, 0, h - 1), ops.clip(x1, 0, h - 1)
    y0, y1 = ops.clip(y0, 0, w - 1), ops.clip(y1, 0, w - 1)

    Ia, Ib = input[:, :, x0, :][:, :, :, y0], input[:, :, x1, :][:, :, :, y0]
    Ic, Id = input[:, :, x0, :][:, :, :, y1], input[:, :, x1, :][:, :, :, y1]

    x1, y1, x0, y0, x, y = map(lambda t: t.astype("float32"), [x1, y1, x0, y0, x, y])
    wa, wb = ops.outer_prod(x1 - x, y1 - y), ops.outer_prod(x1 - x, y - y0)
    wc, wd = ops.outer_prod(x - x0, y1 - y), ops.outer_prod(x - x0, y - y0)

    # TODO -- for some reason, these fails with about 1-3 percent of sum(w) being 1.0
    # total_w = wa + wb + wc + wd
    # ops.np.testing.assert_allclose(total_w.numpy(), ops.ones_like(total_w).numpy(), rtol=1e-5, atol=1e-5)
    return wa * Ia + wb * Ib + wc * Ic + wd * Id
