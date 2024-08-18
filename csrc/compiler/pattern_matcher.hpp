#include "ad_primitives.hpp"
#include "tensor.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pg {

namespace pattern_matcher {

// Base _Pattern Class
class _Pattern {
public:
  virtual bool match(Tensor &t) { throw std::runtime_error("Not implemented"); }
  virtual ~_Pattern() = default;
  _Pattern() = default;
  _Pattern(const _Pattern &) = default;
  _Pattern(_Pattern &&) = default;
  _Pattern &operator=(const _Pattern &) = default;
  std::shared_ptr<Tensor> for_capture_tensor;
};

using Pattern = std::shared_ptr<_Pattern>;

// Input _Pattern Class
class _Input : public _Pattern {
  std::shared_ptr<Tensor> tensor_ptr;

public:
  explicit _Input(std::shared_ptr<Tensor> tensor)
      : tensor_ptr(std::move(tensor)) {}

  bool match(Tensor &t) override {
    *tensor_ptr = t;
    return true;
  }
};

// Base Binary Operation _Pattern Class
class _BinaryOp : public _Pattern {
protected:
  std::shared_ptr<_Pattern> lhs;
  std::shared_ptr<_Pattern> rhs;

public:
  _BinaryOp(std::shared_ptr<_Pattern> lhs, std::shared_ptr<_Pattern> rhs)
      : lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  virtual const std::string get_op_name() const {
    throw std::runtime_error("Not implemented");
  }
  bool match(Tensor &t) override {
    if (t.ad_node()->primitive()->str() != get_op_name() ||
        t.ad_node()->children().size() != 2) {
      return false;
    }
    if (this->for_capture_tensor) {
      *this->for_capture_tensor = t;
    }
    return lhs->match(t.ad_node()->children()[0]) &&
           rhs->match(t.ad_node()->children()[1]);
  }
};

// Base Unary Operation _Pattern Class
class _UnaryOp : public _Pattern {
protected:
  std::shared_ptr<_Pattern> operand;

public:
  explicit _UnaryOp(std::shared_ptr<_Pattern> operand)
      : operand(std::move(operand)) {}

  virtual const std::string get_op_name() const {
    throw std::runtime_error("Not implemented");
  }

  bool match(Tensor &t) override {
    if (t.ad_node()->primitive()->str() != get_op_name() ||
        t.ad_node()->children().size() != 1) {
      return false;
    }
    if (this->for_capture_tensor) {
      *this->for_capture_tensor = t;
    }
    return operand->match(t.ad_node()->children()[0]);
  }
};

// Binary Operation Classes
#define DEFINE_BINARY_OP_CLASS(op)                                             \
  class _##op : public _BinaryOp {                                             \
  public:                                                                      \
    _##op(std::shared_ptr<_Pattern> lhs, std::shared_ptr<_Pattern> rhs)        \
        : _BinaryOp(std::move(lhs), std::move(rhs)) {}                         \
    const std::string get_op_name() const override {                           \
      return std::string(#op);                                                 \
    }                                                                          \
  };

DEFINE_BINARY_OP_CLASS(Add)
DEFINE_BINARY_OP_CLASS(Sub)
DEFINE_BINARY_OP_CLASS(Mul)
DEFINE_BINARY_OP_CLASS(Div)
DEFINE_BINARY_OP_CLASS(Max)
DEFINE_BINARY_OP_CLASS(MatMul)

// Unary Operation Classes
#define DEFINE_UNARY_OP_CLASS(op)                                              \
  class _##op : public _UnaryOp {                                              \
  public:                                                                      \
    explicit _##op(std::shared_ptr<_Pattern> operand)                          \
        : _UnaryOp(std::move(operand)) {}                                      \
    const std::string get_op_name() const override {                           \
      return std::string(#op);                                                 \
    }                                                                          \
  };

DEFINE_UNARY_OP_CLASS(Exp)
DEFINE_UNARY_OP_CLASS(Log)
DEFINE_UNARY_OP_CLASS(Broadcast)
DEFINE_UNARY_OP_CLASS(Permute)
DEFINE_UNARY_OP_CLASS(Reshape)
DEFINE_UNARY_OP_CLASS(Sum)
DEFINE_UNARY_OP_CLASS(MaxReduce)
DEFINE_UNARY_OP_CLASS(Im2Col)

// Factory Functions
std::shared_ptr<_Pattern> Input(std::shared_ptr<Tensor> tensor) {
  return std::make_shared<_Input>(std::move(tensor));
}

std::shared_ptr<_Pattern> Add(std::shared_ptr<_Pattern> lhs,
                              std::shared_ptr<_Pattern> rhs) {
  return std::make_shared<_Add>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> Sub(std::shared_ptr<_Pattern> lhs,
                              std::shared_ptr<_Pattern> rhs) {
  return std::make_shared<_Sub>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> Mul(std::shared_ptr<_Pattern> lhs,
                              std::shared_ptr<_Pattern> rhs) {
  return std::make_shared<_Mul>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> Div(std::shared_ptr<_Pattern> lhs,
                              std::shared_ptr<_Pattern> rhs) {
  return std::make_shared<_Div>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> MatMul(std::shared_ptr<_Pattern> lhs,
                                 std::shared_ptr<_Pattern> rhs) {
  return std::make_shared<_MatMul>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> Max(std::shared_ptr<_Pattern> lhs,
                              std::shared_ptr<_Pattern> rhs) {
  return std::make_shared<_Max>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> Exp(std::shared_ptr<_Pattern> operand) {
  return std::make_shared<_Exp>(std::move(operand));
}

std::shared_ptr<_Pattern> Log(std::shared_ptr<_Pattern> operand) {
  return std::make_shared<_Log>(std::move(operand));
}

std::shared_ptr<_Pattern> Broadcast(std::shared_ptr<_Pattern> operand) {
  return std::make_shared<_Broadcast>(std::move(operand));
}

std::shared_ptr<_Pattern> Permute(std::shared_ptr<_Pattern> operand) {
  return std::make_shared<_Permute>(std::move(operand));
}

std::shared_ptr<_Pattern> Reshape(std::shared_ptr<_Pattern> operand) {
  return std::make_shared<_Reshape>(std::move(operand));
}

std::shared_ptr<_Pattern> Sum(std::shared_ptr<_Pattern> operand) {
  return std::make_shared<_Sum>(std::move(operand));
}

std::shared_ptr<_Pattern> MaxReduce(std::shared_ptr<_Pattern> operand) {
  return std::make_shared<_MaxReduce>(std::move(operand));
}

std::shared_ptr<_Pattern> Im2Col(std::shared_ptr<_Pattern> operand) {
  return std::make_shared<_Im2Col>(std::move(operand));
}

// overwrite << of shared_ptr<_Pattern>
std::shared_ptr<_Pattern> &operator<<(std::shared_ptr<_Pattern> p,
                                      std::shared_ptr<Tensor> t) {
  p->for_capture_tensor = t;
  return p;
}
} // namespace pattern_matcher

} // namespace pg
