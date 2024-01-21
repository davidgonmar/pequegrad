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
