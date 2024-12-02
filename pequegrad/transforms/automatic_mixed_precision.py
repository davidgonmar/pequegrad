from pequegrad.backend.c import compile, clone_graph, Tensor, dt  # noqa
import pequegrad.ops as ops
from pequegrad.transforms.compile import jit, make_pattern_matcher, make_pattern

# everything false
jit_opts_dict = {
    "remove_useless_copy": False,
    "remove_useless_broadcast": False,
    "remove_useless_astype": False,
    "recursive_fused_linear": False,
    "recursive_conv2d": False,
    "recursive_pooling2d": False,
    "recursive_conv2d_vjp_weight": False,
    "recursive_conv2d_vjp_input": False,
    "recursive_local_response_normalization": False,
    "recursive_lrn_vjp_input": False,
    "recursive_max_pooling2d_backward": False,
    "hoist_broadcasts": False,
    "common_subexpr_elim": False,
    "fuser": False,
    "experimental_toposort_optim": False,
}


def not_f16(x):
    return x.dtype != dt.float16


def f16(x):
    return x.dtype == dt.float16


def f32(x):
    return x.dtype == dt.float32


# MATMUL AMP
def matmul_pattern_matcher_fn(a, b):
    return ops.matmul(a, b)


def matmul_pattern_converter(inps):
    a, b = inps
    a, b = a.astype("float16"), b.astype("float16")
    return ops.matmul(a, b).astype("float32")


matmul_pattern_matcher = make_pattern_matcher(
    matmul_pattern_matcher_fn, [(16, 16), (16, 16)], match_inps={0: not_f16, 1: not_f16}
)
matmul_pattern = make_pattern(
    "matmul_amp", matmul_pattern_matcher, matmul_pattern_converter, True
)


# CROSS ENTROPY AMP (make sure not to use float16)
def cross_entropy_loss_probs_pattern_fn(logits, probs):
    return ops.cross_entropy_loss_probs(
        logits.astype("float16"), probs.astype("float16")
    ).astype("float32")


cross_entropy_loss_probs_pattern_matcher = make_pattern_matcher(
    cross_entropy_loss_probs_pattern_fn,
    [(16, 16), (16, 16)],
    match_inps={0: f16, 1: f16},
)


def cross_entropy_loss_probs_pattern_converter(inps):
    return ops.cross_entropy_loss_probs(inps[0], inps[1])


cross_entropy_loss_probs_pattern = make_pattern(
    "cross_entropy_loss_probs_amp",
    cross_entropy_loss_probs_pattern_matcher,
    cross_entropy_loss_probs_pattern_converter,
    True,
)


# GENERAL (elimnate redundant f32->f16->f32 or vice versa)
def astype_pattern_matcher_fn_1(x):
    tof32 = x.astype("float32")
    back = tof32.astype("float16")
    return back


def astype_pattern_matcher_fn_2(x):
    tof16 = x.astype("float16")
    back = tof16.astype("float32")
    return back


astype_pattern_matcher_1 = make_pattern_matcher(
    astype_pattern_matcher_fn_1, [(16, 16)], match_inps={0: f32}
)
astype_pattern_matcher_2 = make_pattern_matcher(
    astype_pattern_matcher_fn_2, [(16, 16)], match_inps={0: f16}
)


def astype_pattern_converter(inps):
    return inps[0]


astype_pattern_1 = make_pattern(
    "astype_f32_f16_f32", astype_pattern_matcher_1, astype_pattern_converter, True
)
astype_pattern_2 = make_pattern(
    "astype_f16_f32_f16", astype_pattern_matcher_2, astype_pattern_converter, True
)


def bias_relu_add_pattern_fn(xw, b):
    return (xw + b).relu()


bias_relu_add_pattern_matcher = make_pattern_matcher(
    bias_relu_add_pattern_fn, [(16, 16), (16,)], match_inps={0: not_f16, 1: not_f16}
)


def bias_relu_add_pattern_converter(inps):
    xw, b = inps[0].astype("float16"), inps[1].astype("float16")
    return (xw + b).relu().astype("float32")


bias_relu_add_pattern = make_pattern(
    "bias_relu_add_amp",
    bias_relu_add_pattern_matcher,
    bias_relu_add_pattern_converter,
    True,
)

patterns = [
    matmul_pattern,
    bias_relu_add_pattern,
    cross_entropy_loss_probs_pattern,
    astype_pattern_1,
    astype_pattern_2,
]


def amp(f):
    return jit(f, opts=jit_opts_dict, eval_outs=False, custom_patterns=patterns)
