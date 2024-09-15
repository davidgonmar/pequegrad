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
  explicit _UnaryOp(std::shared_ptr<_Pattern> operand,
                    std::shared_ptr<Tensor> tensor)
      : operand(std::move(operand)) {
    for_capture_tensor = std::move(tensor);
  }
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
DEFINE_BINARY_OP_CLASS(Pow)

// Unary Operation Classes
#define DEFINE_UNARY_OP_CLASS(op)                                              \
  class _##op : public _UnaryOp {                                              \
  public:                                                                      \
    explicit _##op(std::shared_ptr<_Pattern> operand)                          \
        : _UnaryOp(std::move(operand)) {}                                      \
    explicit _##op(std::shared_ptr<_Pattern> operand,                          \
                   std::shared_ptr<Tensor> tensor)                             \
        : _UnaryOp(std::move(operand), std::move(tensor)) {}                   \
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

// Fill
class _Scalar : public _Pattern {
  float value;

public:
  explicit _Scalar(float value) : value(value) {}

  bool match(Tensor &t) override {
    if (t.ad_node()->primitive()->str().find("Fill") != 0 &&
        ((t.ad_node()->primitive()->str().find("Broadcast") != 0) ||
         (t.ad_node()->children()[0].ad_node()->primitive()->str().find(
              "Fill") != 0))) {
      return false;
    }

    // if it is broadcast, fill is the child;
    if (t.ad_node()->primitive()->str().find("Broadcast") == 0) {
      auto fill = dynamic_cast<Fill &>(
          *t.ad_node()->children()[0].ad_node()->primitive());
      if (fill.value() != value) {
        return false;
      }

      if (this->for_capture_tensor) {
        *this->for_capture_tensor = t;
      }
      return true;
    }

    auto fill = dynamic_cast<Fill &>(*t.ad_node()->primitive());
    if (fill.value() != value) {
      return false;
    }

    if (this->for_capture_tensor) {
      *this->for_capture_tensor = t;
    }

    return true;
  }
};

class _ScalarApprox : public _Pattern {
  float value;

public:
  explicit _ScalarApprox(float value) : value(value) {}

  bool match(Tensor &t) override {
    if (t.ad_node()->primitive()->str().find("Fill") != 0 &&
        ((t.ad_node()->primitive()->str().find("Broadcast") != 0) ||
         (t.ad_node()->children()[0].ad_node()->primitive()->str().find(
              "Fill") != 0))) {
      return false;
    }
    // if it is broadcast, fill is the child;
    if (t.ad_node()->primitive()->str().find("Broadcast") == 0) {
      auto fill = dynamic_cast<Fill &>(
          *t.ad_node()->children()[0].ad_node()->primitive());
      if (std::abs(fill.value() - value) > 1e-3) {
        return false;
      }

      if (this->for_capture_tensor) {
        *this->for_capture_tensor = t;
      }
      return true;
    }

    auto fill = dynamic_cast<Fill &>(*t.ad_node()->primitive());
    if (std::abs(fill.value() - value) > 1e-3) {
      return false;
    }

    if (this->for_capture_tensor) {
      *this->for_capture_tensor = t;
    }

    return true;
  }
};
// Factory Functions
std::shared_ptr<_Pattern> Input(std::shared_ptr<Tensor> tensor) {
  return std::make_shared<_Input>(std::move(tensor));
}

std::shared_ptr<_Pattern> Add(std::shared_ptr<_Pattern> lhs,
                              std::shared_ptr<_Pattern> rhs) {
  return std::make_shared<_Add>(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> Pow(std::shared_ptr<_Pattern> lhs,
                              std::shared_ptr<_Pattern> rhs) {
  return std::make_shared<_Pow>(std::move(lhs), std::move(rhs));
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
std::shared_ptr<_Pattern> Im2Col(std::shared_ptr<_Pattern> operand,
                                 std::shared_ptr<Tensor> tensor) {
  return std::make_shared<_Im2Col>(std::move(operand), std::move(tensor));
}
// overwrite << of shared_ptr<_Pattern>
std::shared_ptr<_Pattern> &operator<<(std::shared_ptr<_Pattern> p,
                                      std::shared_ptr<Tensor> t) {
  p->for_capture_tensor = std::move(t);
  return p;
}

std::shared_ptr<_Pattern> Scalar(float value) {
  return std::make_shared<_Scalar>(value);
}

// Overloaded binary ops

std::shared_ptr<_Pattern> operator+(std::shared_ptr<_Pattern> lhs,
                                    std::shared_ptr<_Pattern> rhs) {
  return Add(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> operator-(std::shared_ptr<_Pattern> lhs,
                                    std::shared_ptr<_Pattern> rhs) {
  return Sub(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> operator*(std::shared_ptr<_Pattern> lhs,
                                    std::shared_ptr<_Pattern> rhs) {
  return Mul(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> operator/(std::shared_ptr<_Pattern> lhs,
                                    std::shared_ptr<_Pattern> rhs) {
  return Div(std::move(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> operator^(std::shared_ptr<_Pattern> lhs,
                                    std::shared_ptr<_Pattern> rhs) {
  return Pow(std::move(lhs), std::move(rhs));
}

// with floats, it is a scalar approx

std::shared_ptr<_Pattern> ScalarApprox(float value) {
  return std::make_shared<_ScalarApprox>(value);
}

std::shared_ptr<_Pattern> operator+(std::shared_ptr<_Pattern> lhs, float rhs) {
  return Add(std::move(lhs), ScalarApprox(rhs));
}

std::shared_ptr<_Pattern> operator-(std::shared_ptr<_Pattern> lhs, float rhs) {
  return Sub(std::move(lhs), ScalarApprox(rhs));
}

std::shared_ptr<_Pattern> operator*(std::shared_ptr<_Pattern> lhs, float rhs) {
  return Mul(std::move(lhs), ScalarApprox(rhs));
}

std::shared_ptr<_Pattern> operator/(std::shared_ptr<_Pattern> lhs, float rhs) {
  return Div(std::move(lhs), ScalarApprox(rhs));
}

std::shared_ptr<_Pattern> operator+(float lhs, std::shared_ptr<_Pattern> rhs) {
  return Add(ScalarApprox(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> operator-(float lhs, std::shared_ptr<_Pattern> rhs) {
  return Sub(ScalarApprox(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> operator*(float lhs, std::shared_ptr<_Pattern> rhs) {
  return Mul(ScalarApprox(lhs), std::move(rhs));
}

std::shared_ptr<_Pattern> operator/(float lhs, std::shared_ptr<_Pattern> rhs) {

  return Div(ScalarApprox(lhs), std::move(rhs));
}

// pow
std::shared_ptr<_Pattern> operator^(std::shared_ptr<_Pattern> lhs, float rhs) {
  return Pow(std::move(lhs), ScalarApprox(rhs));
}

std::shared_ptr<_Pattern> operator^(float lhs, std::shared_ptr<_Pattern> rhs) {
  return Pow(ScalarApprox(lhs), std::move(rhs));
}
/*
def tanh(self):
  # 2 / (1 + torch.exp(-2 * x)) - 1
  return 2 / (1 + pg.exp(-2 * self)) - 1
*/

std::shared_ptr<_Pattern> Tanh(std::shared_ptr<_Pattern> operand) {
  return Div(Scalar(2), Add(Exp(Mul(operand, Scalar(-2))), Scalar(1))) -
         Scalar(1);
}

} // namespace pattern_matcher

} // namespace pg
