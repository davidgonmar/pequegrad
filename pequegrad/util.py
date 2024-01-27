import numpy as np
from typing import Tuple


def im2col(x: np.ndarray, kernel_shape: Tuple[int, int], stride: int = 1) -> np.ndarray:
    """
    Unfold a numpy array to a 3D array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
    It is equivalent to im2col transposed.

    Args:
        x: Input array of shape (batch_size, n_channels, x_h, x_w)
        kernel_shape: Kernel shape (k_h, k_w)
        stride: Stride (default: 1)

    Returns:
        Unfolded array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))

    """

    batch_size, in_channels, x_h, x_w = x.shape
    k_h, k_w = kernel_shape
    out_h = (x_h - k_h) // stride + 1
    out_w = (x_w - k_w) // stride + 1

    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"Invalid kernel size ({k_h}, {k_w}) or stride ({stride}) for input dimensions ({x_h}, {x_w})"
        )

    cols = np.zeros((batch_size, in_channels * k_h * k_w, out_h * out_w))

    for i in range(0, out_h):
        start_i = i * stride
        end_i = start_i + k_h
        for j in range(0, out_w):
            start_j = j * stride
            end_j = start_j + k_w
            cols[:, :, i * out_w + j] = x[:, :, start_i:end_i, start_j:end_j].reshape(
                batch_size, -1
            )

    return cols


def col2im(
    unfolded: np.ndarray,
    kernel_shape: Tuple[int, int],
    output_shape: Tuple[int, int],
    stride: int = 1,
):
    """
    Fold a 3D array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
    It is equivalent to col2im transposed.

    Args:
        unfolded: Unfolded array of shape (batch_size, k_h * k_w * n_channels, (x_h - k_h + 1) * (x_w - k_w + 1))
        kernel_shape: Kernel shape (k_h, k_w)
        output_shape: Output shape (x_h, x_w)
        stride: Stride (default: 1)

    Returns:
        Folded array of shape (batch_size, n_channels, x_h, x_w)

    """
    assert (
        len(unfolded.shape) == 3
    ), "unfolded must have 3 dimensions: (batch, k_h * k_w * n_channels, (out_h - k_h + 1) * (out_w - k_w + 1)), got shape {}".format(
        unfolded.shape
    )

    k_h, k_w = kernel_shape
    out_h, out_w = output_shape
    out_channels = unfolded.shape[1] // (k_h * k_w)
    out_batch = unfolded.shape[0]

    out = np.zeros((out_batch, out_channels, out_h, out_w))

    for i in range(0, (out_h - k_h) // stride + 1):
        for j in range(0, (out_w - k_w) // stride + 1):
            col = unfolded[:, :, i * ((out_w - k_w) // stride + 1) + j]
            start_i = i * stride
            end_i = start_i + k_h
            start_j = j * stride
            end_j = start_j + k_w
            out[:, :, start_i:end_i, start_j:end_j] += col.reshape(
                out_batch, out_channels, k_h, k_w
            )

    return out
