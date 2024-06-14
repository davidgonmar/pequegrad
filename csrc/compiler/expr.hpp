#pragma once

#include "ad_primitives.hpp"
#include <cuda.h>
#include <nvrtc.h>

namespace pg {

class AstExpr {
public:
  DType dtype;
  std::string name;
  virtual std::string render() { throw std::runtime_error("Not implemented"); }
  virtual std::string render_idxs() {
    throw std::runtime_error("Not implemented");
  }
};

enum class AstUnaryOp {
  Log,
  Exp,
};

class AstUnaryExpr : public AstExpr {
public:
  AstUnaryOp op;
  std::shared_ptr<AstExpr> child;

  std::string render() override {
    std::string child_str = child->render();
    if (op == AstUnaryOp::Log) {
      if (dtype == DType::Float32) {
        return "logf(" + child_str + ")";
      } else if (dtype == DType::Float64) {
        return "log(" + child_str + ")";
      }
      PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
    } else if (op == AstUnaryOp::Exp) {
      if (dtype == DType::Float32) {
        return "expf(" + child_str + ")";
      } else if (dtype == DType::Float64) {
        return "exp(" + child_str + ")";
      }
      PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
    }
    PG_CHECK_RUNTIME(false, "Unsupported unary op: " +
                                std::to_string(static_cast<int>(op)));
  }

  std::string render_idxs() override { return child->render_idxs(); }
};

enum class AstBinaryOp {
  Add,
  Mul,
  Max,
};

class AstBinaryExpr : public AstExpr {
public:
  AstBinaryOp op;
  std::shared_ptr<AstExpr> lhs;
  std::shared_ptr<AstExpr> rhs;

  std::string render() override {
    std::string lhs_str = lhs->render();
    std::string rhs_str = rhs->render();
    if (op == AstBinaryOp::Add) {
      return lhs_str + " + " + rhs_str;
      PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
    } else if (op == AstBinaryOp::Mul) {
      return lhs_str + " * " + rhs_str;
      PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
    } else if (op == AstBinaryOp::Max) {
      return "fmax(" + lhs_str + ", " + rhs_str + ")";
      PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
    }
    PG_CHECK_RUNTIME(false, "Unsupported binary op: " +
                                std::to_string(static_cast<int>(op)));
  }

  std::string render_idxs() override {
    return lhs->render_idxs() + rhs->render_idxs();
  }
};

class AstLoadExpr : public AstExpr {

public:
  strides_t strides;
  shape_t shape;
  std ::string render() override {
    // We need to calculate the index from the strides
    // idx = blockIdx.x * blockDim.x + threadIdx.x; -> assumed

    // we just render a constant expression into the rendered code
    std::string st = "";
    for (size_t i = 0; i < shape.size(); i++) {
      st += std::to_string(strides[i] / dtype_to_size(dtype)) + " * " + "in_" +
            name + "_idx" + std::to_string(i);
      if (i != shape.size() - 1) {
        st += " + ";
      }
    }
    // if strides is empty (scalar), st = 0
    if (st == "") {
      st = "0";
    }

    return name + "[" + st + "]";
  }
  std::string render_idxs() override {
    // idx = blockIdx.x * blockDim.x + threadIdx.x;
    // basically does in_idx.... = somefn(idx)
    std::string st = "";
    for (size_t i = 0; i < shape.size(); i++) {
      st += "size_t in_" + name + "_idx" + std::to_string(i) + " = " +
            "idx / " + std::to_string(strides[i] / dtype_to_size(dtype)) +
            " % " + std::to_string(shape[i]) + ";\n";
    }
    return st;
  }
};

class AstStoreExpr : public AstExpr {
public:
  shape_t shape;
  strides_t strides;
  std::shared_ptr<AstExpr> value;
  std::string render() override {

    std::string st = "";
    for (size_t i = 0; i < shape.size(); i++) {
      st += std::to_string(strides[i] / dtype_to_size(dtype)) + " * " + "out_" +
            name + "_idx" + std::to_string(i);
      if (i != shape.size() - 1) {
        st += " + ";
      }
    }

    return name + "[" + st + "] = " + value->render() + ";";
  }

  std::string render_idxs() override {
    // same as load expr, expression based on idx
    std::string st = "";
    for (size_t i = 0; i < shape.size(); i++) {
      st += "size_t out_" + name + "_idx" + std::to_string(i) + " = " +
            "idx / " + std::to_string(strides[i] / dtype_to_size(dtype)) +
            " % " + std::to_string(shape[i]) + ";\n";
    }

    return st + value->render_idxs();
  }
};

template <typename Other> bool is(ADPrimitive &primitive) {
  return typeid(primitive) == typeid(Other);
}

template <typename Other> bool is(std::shared_ptr<ADPrimitive> primitive) {
  return typeid(*primitive) == typeid(Other);
}

class CompiledPrimitive : public ADPrimitive {
  std::string _name;
  std::shared_ptr<AstExpr> ast;

  // cache for function pointer
  void *fn_ptr = nullptr;

public:
  CompiledPrimitive(std::string name, std::shared_ptr<AstExpr> ast)
      : _name(name), ast(ast) {}
  std::string str() { return "CompiledPrimitive(" + _name + ")"; }
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_INFER_OUTPUT_SHAPES
  DEFINE_INFER_OUTPUT_DTYPES
};

// returns a list with the inputs (leafs with LOAD operation)
std::shared_ptr<AstExpr> get_ast_expr(Tensor &curr);

std::vector<std::shared_ptr<AstLoadExpr>>
get_leafs(std::shared_ptr<AstExpr> node);

void fuse(Tensor &out);
} // namespace pg