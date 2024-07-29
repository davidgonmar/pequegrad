#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <mutex>

static __global__ void setup_kernel(curandState *state, unsigned long seed,
                                    int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    curand_init(seed, id, 0, &state[id]);
  }
}

class CudaRandomState {
public:
  static CudaRandomState &get_instance(int n = 0) {
    static CudaRandomState instance;
    if (!instance.initialized_ || (n > instance.n_)) {
      instance.initialize(n);
    }
    return instance;
  }

  std::shared_ptr<curandState> get_states() { return states_; }

private:
  CudaRandomState() : n_(0), initialized_(false) {}

  void initialize(int n) {
    if (n == n_) {
      return;
    }
    n_ = n;
    curandState *states;
    CHECK_CUDA(cudaMallocAsync(&states, n * sizeof(curandState), 0));
    states_ = std::shared_ptr<curandState>(
        states, [](curandState *ptr) { CHECK_CUDA(cudaFreeAsync(ptr, 0)); });
    setup_kernel<<<(n + 255) / 1024, 1024>>>(states_.get(), time(NULL), n);
    PG_CUDA_KERNEL_END;
    initialized_ = true;
  }

  // Disable copy constructor and assignment operator
  CudaRandomState(const CudaRandomState &) = delete;
  CudaRandomState &operator=(const CudaRandomState &) = delete;

  int n_;
  std::shared_ptr<curandState> states_;
  bool initialized_;
};
