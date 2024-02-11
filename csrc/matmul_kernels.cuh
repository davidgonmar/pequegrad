#pragma once

__global__ void
matmul_kernel(const float *a, const float *b, float *out, const size_t *a_shape,
              const size_t *b_shape, const size_t a_num_dims,
              const size_t b_num_dims); // assume contiguous memory
