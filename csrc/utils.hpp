#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <vector>

template <typename T, typename... Args>
void PG_CHECK_ARG(T cond, Args... args) {
  if (!cond) {
    std::ostringstream stream;
    (stream << ... << args);
    throw std::invalid_argument(stream.str());
  }
}

template <typename T, typename... Args>
void PG_CHECK_RUNTIME(T cond, Args... args) {
  if (!cond) {
    std::ostringstream stream;
    (stream << ... << args);
    throw std::runtime_error(stream.str());
  }
}

#define CHECK_INPUTS_LENGTH(inputs_vector, expected_length)                    \
  PG_CHECK_ARG(inputs_vector.size() == expected_length, "Expected ",           \
               expected_length, " inputs, got ", inputs_vector.size())

#define CHECK_OUTPUTS_LENGTH(outputs_vector, expected_length)                  \
  PG_CHECK_ARG(outputs_vector.size() == expected_length, "Expected ",          \
               expected_length, " outputs, got ", outputs_vector.size())

#define CHECK_SAME_SHAPE(tensor1, tensor2)                                     \
  PG_CHECK_ARG(tensor1.shape() == tensor2.shape(),                             \
               "Expected tensors to have the same shape")

template <typename T> std::string vec_to_string(const std::vector<T> &vec) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    ss << vec[i];
    if (i < vec.size() - 1) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}