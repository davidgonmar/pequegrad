__global__ void fill_kernel(float *a, int n, float val) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = val;
  }
}