#pragma once
#include <string>

std::string cuda_ker_header = R"(
// types stride_t and size_t
typedef unsigned long long stride_t;
typedef unsigned long long size_t;


template <typename T>
__device__ int get_idx_from_strides(const size_t *shape,
                                    const stride_t *strides,
                                    const size_t num_dims, const int abs_idx) {
  int tmp_idx = abs_idx;
  int idx = 0;
  for (int d = num_dims - 1; d >= 0; d--) {
    int curr_dim = tmp_idx % shape[d]; // 'how much of dimension d'
    idx += strides[d] * curr_dim;
    tmp_idx /= shape[d];
  }
  return idx / sizeof(T); // strides are in bytes
}

static inline __device__ int get_max_idx(const size_t *shape,
                                         const size_t num_dims) {
  int accum = 1;
  for (int d = 0; d < num_dims; d++) {
    accum *= shape[d];
  }
  return accum;
}
)";

std::string get_x_gid() {
  return "int idx_x = blockIdx.x * blockDim.x + threadIdx.x;";
}

std::string render_kernel_file(const std::string &kernel_name,
                               const std::string &kernel_code,
                               const std::string &inputs_str) {
  std::string file;
  file += cuda_ker_header;
  file += "extern \"C\" {\n";
  file += "__global__ void " + kernel_name + "(" + inputs_str + ") {\n";
  file += kernel_code;
  file += "}\n";
  file += "}\n";
  return file;
}
std::string render_guard(const std::string &idx, const std::string &expr) {
  return "if (" + idx + " >= " + expr + ") { return; }\n";
}