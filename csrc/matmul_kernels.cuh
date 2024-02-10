#pragma once

__global__ void matmul_kernel(const float *a, const float *b, float *out,
                             const int size1, const int sizemid,
                             const int size2);