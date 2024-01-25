import numpy as np
from typing import Tuple


def unfold_numpy_array(x: np.ndarray, kernel_shape: Tuple[int, int]):
    k_h, k_w = kernel_shape
    assert x.ndim == 4, "only accepting batched arrays of ndim 4"
    batch, chann, m, n = x.shape

    # output size = (batch, height of unfolded matrix, flattened kernel (including channels))
    xx = np.zeros((batch, (m - k_h + 1) * (n - k_w + 1), chann * k_h * k_w))
    row = 0
    for i in range(m - k_h + 1):
        for j in range(n - k_w + 1):
            # we will be putting in xx the flattened convolution 'steps' for each position of the filter.
            # all the channels for one position will be included in the flattened array
            # so each xx's row represents a flattened 'step' that includes all channel dimensions
            # batch dimension will be kept in the first dimension
            xx[:, row, :] = x[:, :, i : i + k_h, j : j + k_w].reshape(batch, -1)
            # this is the same as iterating over the batches and doing the following:
            # xx[b, row, :] = x[b, :, i : i + k_h, j : j + k_w].flatten()
            # but is vectorized and faster
            row += 1

    return xx


def fold_numpy_array(
    xx: np.ndarray, kernel_shape: Tuple[int, int], input_shape: Tuple[int, int]
):
    # xx shape = (batch, height of unfolded mat, flattened kernel)
    assert xx.ndim == 3, "only accepting batched arrays of ndim 3"
    # first dimension is batch
    # second dimension is the flattened convolution 'steps' for each position of the filter,
    # so its shape should be (m - k_h + 1) * (n - k_w + 1)
    assert xx.shape[1] == (input_shape[0] - kernel_shape[0] + 1) * (
        input_shape[1] - kernel_shape[1] + 1
    ), "shape mismatch, got {} but expected {}".format(
        xx.shape[1],
        (input_shape[0] - kernel_shape[0] + 1) * (input_shape[1] - kernel_shape[1] + 1),
    )

    # third dimension is the flattened kernel (including channels)
    k_h, k_w = kernel_shape
    chann = xx.shape[2] // (k_h * k_w)
    m, n = input_shape
    batch = xx.shape[0]

    #    [(5, 1, 10, 5), (3, 1, 5, 5)],
    # xx shape = (batch, height of unfolded mat, k_h * k_w * chann)
    out = np.zeros((batch, chann, m, n))

    # iterate over the number of windows
    for i in range(m - k_h + 1):
        for j in range(n - k_w + 1):
            row = xx[:, (n - k_w + 1) * i + j, :]

            # batch, chann, m, n
            out[:, :, i : i + k_h, j : j + k_w] += row.reshape(batch, chann, k_h, k_w)

    return out
