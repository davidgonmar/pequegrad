#pragma once

#include "cuda_fp16.h"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>

enum class DType {
  Int32,
  Float16,
  Float32,
  Float64,
};

std::string dtype_to_string(DType dtype);

DType max_dtype(DType a, DType b);
DType promote_dtype(DType a, DType b);
size_t dtype_to_size(DType dtype);

template <typename T> DType dtype_from_pytype() {
  if (std::is_same<T, int>::value) {
    return DType::Int32;
  } else if (std::is_same<T, long>::value) {
    return DType::Int32;
  } else if (std::is_same<T, unsigned long>::value) {
    return DType::Int32;
  } else if (std::is_same<T, float>::value) {
    return DType::Float32;
  } else if (std::is_same<T, double>::value) {
    return DType::Float64;
  } else if (std::is_same<T, half>::value) {
    return DType::Float16;

  } else {
    throw std::runtime_error("Unknown dtype: " + std::string(typeid(T).name()));
  }
}

template <typename T> DType dtype_from_cpptype() {
  if (std::is_same<T, int>::value) {
    return DType::Int32;
  } else if (std::is_same<T, float>::value) {
    return DType::Float32;
  } else if (std::is_same<T, double>::value) {
    return DType::Float64;
  } else if (std::is_same<T, half>::value) {
    return DType::Float16;

  } else {
    throw std::runtime_error("Unknown dtype: " + std::string(typeid(T).name()));
  }
}

DType dtype_from_string(const std::string &dtype_str);