#include "ad_primitives.hpp"
#include <memory>

namespace pg {
namespace ir {

template <typename Other> bool is(ADPrimitive &primitive) {
  return typeid(primitive) == typeid(Other);
}

template <typename Other> bool is(std::shared_ptr<ADPrimitive> primitive) {
  return typeid(*primitive) == typeid(Other);
}

template <typename Casted> Casted as(ADPrimitive &primitive) {
  return dynamic_cast<Casted>(primitive);
}

template <typename Casted>
std::shared_ptr<Casted> as(std::shared_ptr<ADPrimitive> primitive) {
  return std::dynamic_pointer_cast<Casted>(primitive);
}

// Linear IR (Intermediate Representation)
class BaseExpr {
public:
  std::string name;
  virtual ~BaseExpr() = default;
  virtual std::string expr_str() { return "BaseExpr"; }
};

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
  DType dtype;
  double value; // will be casted to the correct type
  std::string expr_str() override { return "ImmExpr"; }
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
  int index;
  std::string expr_str() override { return "ArgExpr"; }
};

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
  std::shared_ptr<BaseExpr> child;
  std::string expr_str() override { return "UnaryExpr"; }
};

enum class BinaryOpKind { Add, Mul, Max, Sub, Div, Gt, Lt, Mod };

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
  std::string expr_str() override { return "BinaryExpr"; }
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
*/

class ForStartExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> start;
  std::shared_ptr<BaseExpr> end;
  std::shared_ptr<BaseExpr> step;
  std::string expr_str() override { return "ForStartExpr"; }
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

class ForEndExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> for_start;
  std::string expr_str() override { return "ForEndExpr"; }
};

/*Represents a special cuda operation to get the thread id
{
    axis: 0
}
CODE:
    int x = threadIdx.x;
*/

class ThreadIdxExpr : public BaseExpr {
public:
  int axis; // 0 (x), 1 (y), 2 (z)
  std::string expr_str() override { return "ThreadIdxExpr"; }
};

/*Represents a special cuda operation to get the block id
{
    axis: 0
}
CODE:
    int x = blockIdx.x;
*/

class BlockIdxExpr : public BaseExpr {
public:
  int axis; // 0 (x), 1 (y), 2 (z)
  std::string expr_str() override { return "BlockIdxExpr"; }
};

/*Represents a special cuda operation to get the block dimension
{
    axis: 0
}

CODE:
    int x = blockDim.x;
*/

class BlockDimExpr : public BaseExpr {
public:
  int axis; // 0 (x), 1 (y), 2 (z)
  std::string expr_str() override { return "BlockDimExpr"; }
};

/*Represents a load operation
{
    name: x
    child: {
        ...,
        name: y
    }
    idx: {
        ...,
        name: z

    }
}
CODE:
    x = y[z];
*/

class LoadExpr : public BaseExpr {
public:
  std::string name;
  std::shared_ptr<BaseExpr> child;
  std::shared_ptr<BaseExpr> idx;
  DType dtype;
  std::string expr_str() override { return "LoadExpr"; }
};

/*Represents a store operation
{
    name: x
    child: {
        ...,
        name: y
    }
    idx: {
        ...,
        name: z

    }
}
CODE:
    x[z] = y;
*/

class StoreExpr : public BaseExpr {
public:
  std::string name;
  std::shared_ptr<BaseExpr> ptr;
  std::shared_ptr<BaseExpr> idx;
  std::shared_ptr<BaseExpr> value;
  DType dtype;
  std::string expr_str() override { return "StoreExpr"; }
};

using ir_t = std::vector<std::shared_ptr<BaseExpr>>;
std::vector<std::shared_ptr<BaseExpr>> graph_to_ir(Tensor &out,
                                                   std::vector<Tensor> &inputs);

std::string ir_to_string(ir_t &ir);
std::string ir_to_cuda(ir_t &ir);

template <typename T> bool is(BaseExpr &expr) {
  return typeid(expr) == typeid(T);
}

template <typename T> bool is(std::shared_ptr<BaseExpr> expr) {
  return typeid(*expr) == typeid(T);
}

template <typename T> T as(BaseExpr &expr) { return dynamic_cast<T>(expr); }

template <typename T> std::shared_ptr<T> as(std::shared_ptr<BaseExpr> &ir) {
  return std::dynamic_pointer_cast<T>(ir);
}

class NameDatabase {
private:
  int tmp_counter_ = 0;
  int const_counter_ = 0;
  int arg_counter_ = 0;

public:
  std::string get_unique_name_tmp() {
    return "tmp" + std::to_string(tmp_counter_++);
  }
  std::string get_unique_name_const() {
    return "const" + std::to_string(const_counter_++);
  }
  std::string get_unique_name_arg() {
    return "arg" + std::to_string(arg_counter_++);
  }
  std::string get_idx_name(BlockDimExpr &expr) {
    return "bdim" + std::to_string(expr.axis);
  }
  std::string get_idx_name(ThreadIdxExpr &expr) {
    return "tidx" + std::to_string(expr.axis);
  }
  std::string get_idx_name(BlockIdxExpr &expr) {
    return "bidx" + std::to_string(expr.axis);
  }

  std::string get_name(std::shared_ptr<BaseExpr> expr) {
    if (is<BlockDimExpr>(expr)) {
      return get_idx_name(*as<BlockDimExpr>(expr).get());
    }
    if (is<ThreadIdxExpr>(expr)) {
      return get_idx_name(*as<ThreadIdxExpr>(expr).get());
    }
    if (is<BlockIdxExpr>(expr)) {
      return get_idx_name(*as<BlockIdxExpr>(expr).get());
    }
    if (is<ImmExpr>(expr)) {
      return get_unique_name_const();
    }
    if (is<BinaryExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<UnaryExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<ArgExpr>(expr)) {
      return get_unique_name_arg();
    }
    if (is<ForStartExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<ForEndExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<LoadExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<StoreExpr>(expr)) {
      return get_unique_name_tmp();
    }

    PG_CHECK_RUNTIME(false, "Unsupported expression: " + expr->expr_str());
  }
};

} // namespace ir

} // namespace pg