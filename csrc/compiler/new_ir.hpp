#include "ad_primitives.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pg {
namespace newir {
class BaseExpr {
public:
  virtual ~BaseExpr() = default;
  virtual std::string expr_str() { return "BaseExpr"; }
};
using ir_item_t = std::shared_ptr<BaseExpr>;

using ir_t = std::vector<std::shared_ptr<BaseExpr>>;
DType getdt(ir_item_t it);

/*Represents immediate assignments like
{
    dtype: Float32,
    value: 3.0
}
CODE:
    float32 x = 3.0;
*/
class ImmExpr : public BaseExpr {
public:
  DType dtype = DType::Float32;
  double value; // will be casted to the correct type
  std::string expr_str() override { return "ImmExpr"; }
  ImmExpr(DType dtype, double value) : dtype(dtype), value(value) {}
};

/*Represents an argument to the kernel
{
    index: 0
    dtype: Float32
    name: x
}
CODE:
    kernel_name(float32 *x, ...)
*/
class ArgExpr : public BaseExpr {
public:
  DType dtype;
  int idx;
  std::string expr_str() override { return "ArgExpr"; }
  ArgExpr(DType dtype, int idx) : dtype(dtype), idx(idx) {}
};

using ir_arg_t = std::shared_ptr<ArgExpr>;
enum class UnaryOpKind { Log, Exp };

/*Represents unary operations like
{
    op: Log,
    child: {
        ...,
        name: y
    }
}
CODE:
    x = log(y);
*/
class UnaryExpr : public BaseExpr {
public:
  UnaryOpKind op;
  std::shared_ptr<BaseExpr> src;
  DType dtype;
  std::string expr_str() override { return "UnaryExpr"; }
};

enum class BinaryOpKind { Add, Mul, Max, Sub, Div, Gt, Lt, Mod, Eq, Pow };

/*Represents binary operations like
{
    op: Add,
    lhs: {
        ...,
        name: y
    },
    rhs: {
        ...,
        name: z
    }
}
CODE:
    x = y + z;
*/

class BinaryExpr : public BaseExpr {
public:
  BinaryOpKind op;
  std::shared_ptr<BaseExpr> lhs;
  std::shared_ptr<BaseExpr> rhs;
  DType dtype;
  std::string expr_str() override {
    std::map<BinaryOpKind, std::string> op_to_str = {
        {BinaryOpKind::Add, "Add"}, {BinaryOpKind::Mul, "Mul"},
        {BinaryOpKind::Max, "Max"}, {BinaryOpKind::Sub, "Sub"},
        {BinaryOpKind::Div, "Div"}, {BinaryOpKind::Gt, "Gt"},
        {BinaryOpKind::Lt, "Lt"},   {BinaryOpKind::Mod, "Mod"},
        {BinaryOpKind::Eq, "Eq"},   {BinaryOpKind::Pow, "Pow"}};
    return "BinaryExpr";
  }
  BinaryExpr(BinaryOpKind op, std::shared_ptr<BaseExpr> lhs,
             std::shared_ptr<BaseExpr> rhs)
      : op(op), lhs(lhs), rhs(rhs) {
    // infer dtype (must match for both)
    auto dt1 = getdt(lhs);
    auto dt2 = getdt(rhs);
    if (dt1 != dt2) {
      throw std::runtime_error(
          "BinaryExpr: dtypes must match: " + dtype_to_string(dt1) +
          " != " + dtype_to_string(dt2));
    }
    dtype = dt1;
  }
  BinaryExpr() = default;
};

enum class AssignOpKind { Add, Max };

// rendered like dst = src (so name is useless)
class AssignExpr : public BaseExpr {
public:
  AssignOpKind op;
  std::shared_ptr<BaseExpr> dst;
  std::shared_ptr<BaseExpr> src;
  DType dtype;
  std::string expr_str() override { return "AccumExpr"; }
};

enum class TernaryOpKind { Where };

/*Represents ternary operations like
{
    op: Where,
    cond: {
        ...,
        name: y
    },
    lhs: {
        ...,
        name: z
    },
    rhs: {
        ...,
        name: w
    }
}
CODE:
    x = y ? z : w;
*/

class TernaryExpr : public BaseExpr {
public:
  TernaryOpKind op;
  std::shared_ptr<BaseExpr> first;
  std::shared_ptr<BaseExpr> second;
  std::shared_ptr<BaseExpr> third;
  DType dtype;
  std::string expr_str() override { return "TernaryExpr"; }
};

/*Represents a for loop start
{
    start: {
        ...,
        name: x
    }
    end: {
        ...,
        name: y
    }
    step: {
        ...,
        name: z
    }

}
CODE:
    for (x; x < y; x += z) {
    }
*/

DType getdt(ir_item_t it) {
  std::string str = it->expr_str();
  if (str == "ImmExpr") {
    return std::static_pointer_cast<ImmExpr>(it)->dtype;
  } else if (str == "ArgExpr") {
    return std::static_pointer_cast<ArgExpr>(it)->dtype;
  } else if (str == "UnaryExpr") {
    return std::static_pointer_cast<UnaryExpr>(it)->dtype;
  } else if (str == "BinaryExpr") {
    return std::static_pointer_cast<BinaryExpr>(it)->dtype;
  } else if (str == "AssignExpr") {
    return std::static_pointer_cast<AssignExpr>(it)->dtype;
  } else if (str == "TernaryExpr") {
    return std::static_pointer_cast<TernaryExpr>(it)->dtype;
  } else if (str == "BoundIdxExpr") {
    return DType::Int32;
  } else {
    throw std::runtime_error("getdt: unknown type: " + str);
  }
}
enum class LoopOpKind { Parallel, Sequential };
class LoopExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> start;
  std::shared_ptr<BaseExpr> end;
  std::shared_ptr<BaseExpr> step;
  std::vector<std::shared_ptr<BaseExpr>> body;
  LoopOpKind op;
  std::string expr_str() override { return "LoopExpr"; }
  LoopExpr(LoopOpKind op, std::shared_ptr<BaseExpr> start,
           std::shared_ptr<BaseExpr> end, std::shared_ptr<BaseExpr> step)
      : op(op), start(start), end(end), step(step) {}
};

class BoundIdxExpr : public BaseExpr {
public:
  std::shared_ptr<LoopExpr> loop;
  DType dtype = DType::Int32;
  std::string expr_str() override { return "BoundIdxExpr"; }
  BoundIdxExpr(std::shared_ptr<LoopExpr> loop) : loop(loop) {}
};

/*Represents a for loop end
{
    for_start: {
        ...,
        name: x
    }
}
CODE:
    }
*/

class ReturnExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> value;
  std::string expr_str() override { return "ReturnExpr"; }
};

class FunctionExpr : public BaseExpr {
public:
  std::string name;
  std::vector<std::shared_ptr<ArgExpr>> args;
  std::vector<std::shared_ptr<BaseExpr>> body;
  std::string expr_str() override { return "FunctionExpr"; }

  FunctionExpr(std::string name, std::vector<std::shared_ptr<ArgExpr>> args)
      : name(name), args(args) {}
};

class LoadExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> src;
  std::shared_ptr<BaseExpr> idx;
  std::string expr_str() override { return "LoadExpr"; }
  LoadExpr(std::shared_ptr<BaseExpr> src, std::shared_ptr<BaseExpr> idx)
      : src(src), idx(idx) {}
};

class StoreExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> dst;
  std::shared_ptr<BaseExpr> idx;
  std::shared_ptr<BaseExpr> src;

  std::string expr_str() override { return "StoreExpr"; }
  StoreExpr(std::shared_ptr<BaseExpr> dst, std::shared_ptr<BaseExpr> idx,
            std::shared_ptr<BaseExpr> src)
      : dst(dst), idx(idx), src(src) {}
};

// overloads for ops like +, -, *, /, etc
ir_item_t operator+(ir_item_t lhs, ir_item_t rhs) {
  return std::make_shared<BinaryExpr>(BinaryOpKind::Add, lhs, rhs);
}

ir_item_t operator-(ir_item_t lhs, ir_item_t rhs) {
  return std::make_shared<BinaryExpr>(BinaryOpKind::Sub, lhs, rhs);
}

ir_item_t operator*(ir_item_t lhs, ir_item_t rhs) {
  return std::make_shared<BinaryExpr>(BinaryOpKind::Mul, lhs, rhs);
}

ir_item_t operator/(ir_item_t lhs, ir_item_t rhs) {
  return std::make_shared<BinaryExpr>(BinaryOpKind::Div, lhs, rhs);
}

ir_item_t immint(int value) {
  return std::make_shared<ImmExpr>(DType::Int32, value);
}

} // namespace newir
} // namespace pg