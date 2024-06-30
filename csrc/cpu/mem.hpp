#pragma once
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

static inline std::shared_ptr<void> allocate_cpu(const size_t nbytes) {
  char *ptr = new char[nbytes];
  std::shared_ptr<void> shared_ptr(
      ptr, [nbytes](void *ptr) { delete[] static_cast<char *>(ptr); });
  return shared_ptr;
}
