#pragma once
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdexcept>

// pinned means cuda pinned memory
static inline std::shared_ptr<void> allocate_cpu(const size_t nbytes,
                                                 bool pinned) {
  if (pinned) {
    // use cuda host alloc with pinned memory
    void *ptr;
    if (cudaHostAlloc(&ptr, nbytes, cudaHostAllocDefault) != cudaSuccess) {
      throw std::runtime_error("cudaHostAlloc failed");
    }
    return std::shared_ptr<void>(ptr, [](void *p) { cudaFreeHost(p); });
  }
  return std::shared_ptr<void>(new char[nbytes], std::default_delete<char[]>());
}
