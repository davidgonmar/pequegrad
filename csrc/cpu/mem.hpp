#pragma once
#include <cstdlib>
#include <memory>
#include <stdexcept>

static inline std::shared_ptr<void> allocate_cpu(const size_t nbytes) {
  return std::shared_ptr<void>(new char[nbytes], std::default_delete<char[]>());
}