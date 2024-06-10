#pragma once
#include "dtype.hpp"
#include "tensor.hpp"
#include <string>
class Expr {
public:
  std::string expr_name;
  DType dtype;
};