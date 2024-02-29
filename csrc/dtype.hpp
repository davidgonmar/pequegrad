#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>

enum class DType {
  Int32,
  Float32,
  Float64,
};

std::string dtype_to_string(DType dtype);

constexpr DType max_dtype(DType a, DType b);

size_t dtype_to_size(DType dtype);

template <typename T> DType dtype_from_pytype() {
  if (std::is_same<T, int>::value) {
    return DType::Int32;
  } else if (std::is_same<T, float>::value) {
    return DType::Float32;
  } else if (std::is_same<T, double>::value) {
    return DType::Float64;
  } else {
    throw std::runtime_error("Unknown dtype");
  }
}

template <typename T> DType dtype_from_cpptype() {
  if (std::is_same<T, int>::value) {
    return DType::Int32;
  } else if (std::is_same<T, float>::value) {
    return DType::Float32;
  } else if (std::is_same<T, double>::value) {
    return DType::Float64;
  } else {
    throw std::runtime_error("Unknown dtype");
  }
}