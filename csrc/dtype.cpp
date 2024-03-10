#include "dtype.hpp"
#include <stdexcept>
#include <string>

std::string dtype_to_string(DType dtype) {
  switch (dtype) {
  case DType::Int32:
    return "int32";
  case DType::Float32:
    return "float32";
  case DType::Float64:
    return "float64";
  default:
    throw std::runtime_error("Unknown dtype: " +
                             std::to_string(static_cast<int>(dtype)));
  }
}

constexpr DType max_dtype(DType a, DType b) {
  return (static_cast<int>(a) > static_cast<int>(b)) ? a : b;
}

size_t dtype_to_size(DType dtype) {
  switch (dtype) {
  case DType::Int32:
    return sizeof(int);
  case DType::Float32:
    return sizeof(float);
  case DType::Float64:
    return sizeof(double);
  default:
    throw std::runtime_error("Unknown dtype: " +
                             std::to_string(static_cast<int>(dtype)));
  }
}
