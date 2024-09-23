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
  case DType::Float16:
    return "float16";
  default:
    throw std::runtime_error("Unknown dtype: " +
                             std::to_string(static_cast<int>(dtype)));
  }
}

DType max_dtype(DType a, DType b) {
  return (static_cast<int>(a) > static_cast<int>(b)) ? a : b;
}

DType promote_dtype(DType a, DType b) {
  if (a == b) {
    return a;
  }
  return max_dtype(a, b);
}

size_t dtype_to_size(DType dtype) {
  switch (dtype) {
  case DType::Int32:
    return sizeof(int);
  case DType::Float16:
    return sizeof(float) / 2;
  case DType::Float32:
    return sizeof(float);
  case DType::Float64:
    return sizeof(double);
  default:
    throw std::runtime_error("Unknown dtype: " +
                             std::to_string(static_cast<int>(dtype)));
  }
}
DType dtype_from_string(const std::string &dtype_str) {
  if (dtype_str == "int32") {
    return DType::Int32;
  } else if (dtype_str == "float32") {
    return DType::Float32;
  } else if (dtype_str == "float64") {
    return DType::Float64;
  } else if (dtype_str == "float16") {
    return DType::Float16;
  } else {
    throw std::runtime_error("Unknown dtype: " + dtype_str);
  }
}