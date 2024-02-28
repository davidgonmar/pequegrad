#pragma once

template <typename T> __global__ void fill_kernel(T *a, int n, T val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = val;
  }
}