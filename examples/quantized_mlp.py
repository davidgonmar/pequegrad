import pequegrad as pg

# Quantization!

NBITS = 8
size = 8192
num_iterations = 20


def bias_relu_add_pattern_fn(x, w, b):
    return (x @ w + b).relu()


bias_relu_add_pattern_matcher = pg.make_pattern_matcher(
    bias_relu_add_pattern_fn, [(16, 16), (16, 16), (16,)]
)


def bias_add_pattern_fn(x, w, b):
    return x @ w + b


bias_add_pattern_matcher = pg.make_pattern_matcher(
    bias_add_pattern_fn, [(16, 16), (16, 16), (16,)]
)


def quant_weights(w):
    wmaxabs = w.abs().max_reduce()
    wscale = pg.max(wmaxabs, 1e-6)
    w = pg.round(w / wscale * (2 ** (NBITS - 1) - 1))
    w = pg.clip(w, -(2 ** (NBITS - 1)), 2 ** (NBITS - 1) - 1)
    w = w / (2 ** (NBITS - 1) - 1) * wscale
    return w


def bias_relu_add_pattern_converter(inps):
    x, w, b = inps[0], inps[1], inps[2]
    w = quant_weights(w)
    xw = pg.matmul(x, w)
    return (xw + b).relu()


def bias_add_pattern_converter(inps):
    x, w, b = inps[0], inps[1], inps[2]
    w = quant_weights(w)
    xw = pg.matmul(x, w)
    return xw + b


bias_relu_add_pattern = pg.make_pattern(
    "bias_relu_add_quantized",
    bias_relu_add_pattern_matcher,
    bias_relu_add_pattern_converter,
    True,
)

bias_add_pattern = pg.make_pattern(
    "bias_add_quantized",
    bias_add_pattern_matcher,
    bias_add_pattern_converter,
    True,
)

patterns = [
    bias_relu_add_pattern,
    bias_add_pattern,
]


def quantized(f):
    return pg.jit(f, opts=pg.jit_opts_dict, eval_outs=False, custom_patterns=patterns)


def linear_relu(x, w, b):
    def _f(x, w, b):
        l1 = pg.relu(x @ w + b)
        l2 = pg.relu(l1 @ w + b)
        return pg.relu(l2 @ w + b)

    return _f(x, w, b)


if __name__ == "__main__":
    quant_linear_relu = pg.jit(
        quantized(linear_relu),
        opts={"fused_linear_relu": False, "experimental_toposort_optim": False},
    )

    x = pg.Tensor(pg.np.random.randn(size, size) / 90).to("cuda").astype("float32")
    w = pg.Tensor(pg.np.random.randn(size, size) / 90).to("cuda").astype("float32")
    b = pg.Tensor(pg.np.zeros(size)).to("cuda").astype("float32")

    wq = quant_weights(w)
    print(pg.np.abs(wq.numpy() - w.numpy()))

    wb = quant_weights(b)
    print(pg.np.abs(wb.numpy() - b.numpy()))

    res = quant_linear_relu(x, w, b)
    res2 = linear_relu(x, w, b)

    print(res.numpy())
    print(res2.numpy())

    # Compute error
    error = pg.np.abs(res.numpy() - res2.numpy())
    print("Error:", error.mean())
