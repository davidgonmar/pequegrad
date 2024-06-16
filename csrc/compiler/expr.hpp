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
  virtual void propagate_movement_ops() {
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
  void propagate_movement_ops() override { child->propagate_movement_ops(); }
};

enum class AstBinaryOp {
  Add,
  Mul,
  Max,
  Gt,
  Lt,
  Eq,
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
    } else if (op == AstBinaryOp::Gt) {
      return lhs_str + " > " + rhs_str;
      PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
    } else if (op == AstBinaryOp::Lt) {
      return lhs_str + " < " + rhs_str;
      PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
    } else if (op == AstBinaryOp::Eq) {
      return lhs_str + " == " + rhs_str;
      PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
    }
    PG_CHECK_RUNTIME(false, "Unsupported binary op: " +
                                std::to_string(static_cast<int>(op)));
  }

  std::string render_idxs() override {
    return lhs->render_idxs() + rhs->render_idxs();
  }
  void propagate_movement_ops() override {
    lhs->propagate_movement_ops();
    rhs->propagate_movement_ops();
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
    // scalar case
    if (shape.size() == 0) {
      return "";
    }
    // this only calculates the index for each dim
    std::string st = "";
    st += "size_t in_" + name + "_idx" + std::to_string(shape.size() - 1) +
          " = " + "idx" + " % " + std::to_string(shape[shape.size() - 1]) +
          ";\n";
    std::string divisor = "";
    for (int i = shape.size() - 2; i >= 0; i--) {
      divisor += (divisor == "" ? std::to_string(shape[i + 1])
                                : " / " + std::to_string(shape[i + 1]));
      st += "size_t in_" + name + "_idx" + std::to_string(i) + " = " +
            "(idx / " + divisor + ") % " + std::to_string(shape[i]) + ";\n";
    }

    return st;
  }
  void propagate_movement_ops() override {}
};

class AstStoreExpr : public AstExpr {
public:
  shape_t shape;
  strides_t strides;
  std::shared_ptr<AstExpr> value;
  std::string render() override {
    // scalar case
    if (shape.size() == 0) {
      return name + "[0]" + " = " + value->render() + ";";
    }
    std::string st = "";
    for (size_t i = 0; i < shape.size(); i++) {
      st += std::to_string(strides[i] / dtype_to_size(dtype)) + " * " + "out_" +
            name + "_idx" + std::to_string(i);
      if (i != shape.size() - 1) {
        st += " + ";
      }
    }
    if (st == "") {
      st = "0";
    }
    std::string a = value->render();
    return name + "[" + st + "] = " + a + ";";
  }

  std::string render_idxs() override {
    // same as load expr, expression based on idx
    // scalar case
    if (shape.size() == 0) {
      return "";
    }
    std::string st = "";
    st += "size_t out_" + name + "_idx" + std::to_string(shape.size() - 1) +
          " = " + "idx" + " % " + std::to_string(shape[shape.size() - 1]) +
          ";\n";
    std::string divisor = "";
    for (int i = shape.size() - 2; i >= 0; i--) {
      divisor += (divisor == "" ? std::to_string(shape[i + 1])
                                : " / " + std::to_string(shape[i + 1]));
      st += "size_t out_" + name + "_idx" + std::to_string(i) + " = " +
            "(idx / " + divisor + ") % " + std::to_string(shape[i]) + ";\n";
    }
    return st + value->render_idxs();
  }
  void propagate_movement_ops() override { value->propagate_movement_ops(); }
};

class AstConstExpr : public AstExpr {
public:
  double val = 0;
  std::string render() override {
    // cast value to the correct type
    if (dtype == DType::Float32) {
      return "(float)" + std::to_string(val);
    } else if (dtype == DType::Float64) {
      return "(double)" + std::to_string(val);
    }
    PG_CHECK_RUNTIME(false, "Unsupported dtype: " + dtype_to_string(dtype));
  }
  std::string render_idxs() override { return ""; }
  void propagate_movement_ops() override {}
};

enum class AstTernaryOp { Where };
class AstTernaryExpr : public AstExpr {
public:
  AstTernaryOp op;
  std::shared_ptr<AstExpr> first;
  std::shared_ptr<AstExpr> second;
  std::shared_ptr<AstExpr> third;

  std::string render() override {
    std::string first_str = first->render();
    std::string second_str = second->render();
    std::string third_str = third->render();
    if (op == AstTernaryOp::Where) {
      return "(" + first_str + " ? " + second_str + " : " + third_str + ")";
    }
    PG_CHECK_RUNTIME(false, "Unsupported ternary op: " +
                                std::to_string(static_cast<int>(op)));
  }

  std::string render_idxs() override {
    return first->render_idxs() + second->render_idxs() + third->render_idxs();
  }
  void propagate_movement_ops() override {
    first->propagate_movement_ops();
    second->propagate_movement_ops();
    third->propagate_movement_ops();
  }
};

class AstPermuteOp : public AstExpr {
public:
  std::shared_ptr<AstLoadExpr> child;
  std::vector<size_t> permute;
  std::string render() override { return child->render(); }
  std::string render_idxs() override { return child->render_idxs(); }

  void propagate_movement_ops() override {
    // so here we will shuffle the axes of our child
    strides_t strides = child->strides;
    shape_t shape = child->shape;

    // we need to permute the strides and shape
    shape_t new_shape;
    strides_t new_strides;
    for (size_t i = 0; i < permute.size(); i++) {
      new_shape.push_back(shape[permute[i]]);
      new_strides.push_back(strides[permute[i]]);
    }
    child->shape = new_shape;
    child->strides = new_strides;
  }
};

class AstBroadcastOp : public AstExpr {
public:
  std::shared_ptr<AstLoadExpr> child;
  shape_t shape;
  std::string render() override { return child->render(); }
  std::string render_idxs() override { return child->render_idxs(); }
  void propagate_movement_ops() override {
    // we need to broadcast the child to the new shape
    // we need to calculate the new strides
    shape_t new_shape = shape;
    strides_t new_strides(new_shape.size());
    strides_t child_strides = child->strides;
    shape_t child_shape = child->shape;
    for (int i = new_shape.size() - 1; i >= 0; i--) {
      if (new_shape[i] == 1) {
        new_strides[i] = 0;
      } else {
        new_strides[i] = child_strides[i];
      }
    }
    child->shape = new_shape;
    child->strides = new_strides;
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
  std::string _cuda_code;

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

bool fuse(Tensor &out);
} // namespace pg