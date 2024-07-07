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
  bool force_render = false;
  virtual ~BaseExpr() = default;
  virtual std::string expr_str() { return "BaseExpr"; }
};
using ir_item_t = std::shared_ptr<BaseExpr>;

using ir_t = std::vector<std::shared_ptr<BaseExpr>>;

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
  std::shared_ptr<BaseExpr> child;
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
    return "BinaryExpr<" + op_to_str.at(op) + ">";
  }
};

enum class AccumOpKind { Add, Max };

// rendered like lhs = lhs + rhs (so name is useless)
class AccumExpr : public BaseExpr {
public:
  AccumOpKind op;
  std::shared_ptr<BaseExpr> lhs;
  std::shared_ptr<BaseExpr> rhs;
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
*/

class ForStartExpr : public BaseExpr {
public:
  int unroll = 0;
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

class IfStartExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> cond;
  std::string expr_str() override { return "IfExpr"; }
};

class IfEndExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> if_start;
  std::string expr_str() override { return "IfEndExpr"; }
};

class ReturnExpr : public BaseExpr {
public:
  std::shared_ptr<BaseExpr> value;
  std::string expr_str() override { return "ReturnExpr"; }
};

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

class IrBuilderContext {
public:
  std::map<int, ir_arg_t> tid_to_arg;
  std::map<ir_arg_t, ir_t> arg_to_strides;
  std::map<ir_arg_t, ir_t> arg_to_shape;
  std::map<ir_arg_t, ir_t> arg_to_idxs_to_load;
};

std::tuple<std::vector<std::shared_ptr<BaseExpr>>, IrBuilderContext,
           std::vector<bool>>
graph_to_ir(Tensor &out, const std::vector<Tensor> &inputs);

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
  std::map<std::string, int> prefix_counter_;

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
    if (is<IfStartExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<IfEndExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<ReturnExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<TernaryExpr>(expr)) {
      return get_unique_name_tmp();
    }
    if (is<AccumExpr>(expr)) {
      return get_unique_name_tmp();
    }
    PG_CHECK_RUNTIME(false, "Unsupported expression: " + expr->expr_str());
  }

  std::string get_with_prefix(std::string prefix) {
    if (prefix_counter_.find(prefix) == prefix_counter_.end()) {
      prefix_counter_[prefix] = 0;
    }
    return prefix + std::to_string(prefix_counter_[prefix]++);
  }
};

// something like (arg1 * arg2) * arg3
static ir_t binop_chain(ir_t operands, BinaryOpKind op_kind) {
  ir_t res;
  for (int i = 0; i < operands.size(); i++) {
    auto operand = operands[i];
    if (i == 0) {
      res.push_back(operand);
    } else {
      auto new_binop = std::make_shared<BinaryExpr>();
      new_binop->op = op_kind;
      new_binop->lhs = res.back();
      new_binop->rhs = operand;
      res.push_back(new_binop);
    }
  }
  return res;
}

static ir_t get_imm_expr(double value, DType dtype) {
  auto imm = std::make_shared<ImmExpr>();
  imm->dtype = dtype;
  imm->value = value;
  return {imm};
}

// renders Tensor<shape=(A, B, C), strides=(X, Y, Z)>[i, j, k] where 0 <= i < A,
// 0 <= j < B, 0 <= k < C
static ir_t render_load_idxs_for_expr(ir_t load_idxs, ir_t tensor_strides,
                                      ir_arg_t arg) {
  PG_CHECK_RUNTIME(
      load_idxs.size() == tensor_strides.size(),
      "[render_load_idxs_for_expr] load_idxs.size() != tensor_strides.size()");
  ir_t res;
  if (load_idxs.size() == 0) {
    auto load = std::make_shared<LoadExpr>();
    load->child = arg;
    auto zero = std::make_shared<ImmExpr>();
    zero->value = 0;
    res.push_back(zero);
    load->idx = zero;
    load->dtype = arg->dtype;
    res.push_back(load);
    return res;
  }

  std::vector<std::shared_ptr<BaseExpr>> muls;

  for (int i = 0; i < load_idxs.size(); i++) {
    auto mul = std::make_shared<BinaryExpr>();
    mul->op = BinaryOpKind::Mul;
    mul->lhs = load_idxs[i];
    mul->rhs = tensor_strides[i];
    res.push_back(mul);
    muls.push_back(mul);
  }

  // now final expression summing all
  int x = muls.size();
  auto sum = binop_chain(muls, BinaryOpKind::Add);

  res.insert(res.end(), sum.begin(), sum.end());
  // now, do the load
  auto load = std::make_shared<LoadExpr>();
  load->child = arg;
  load->idx = sum.back();
  load->dtype = arg->dtype;
  res.push_back(load);
  return res;
}

// same that for load, but at the end we just do a store
static ir_t render_store_idxs_for_expr(ir_t store_idxs, ir_t tensor_strides,
                                       ir_arg_t arg,
                                       std::shared_ptr<BaseExpr> value) {
  PG_CHECK_RUNTIME(store_idxs.size() == tensor_strides.size(),
                   "[render_store_idxs_for_expr] store_idxs.size() != "
                   "tensor_strides.size()");
  ir_t res;
  if (store_idxs.size() == 0) {
    auto store = std::make_shared<StoreExpr>();
    store->ptr = arg;
    auto zero = std::make_shared<ImmExpr>();
    zero->value = 0;
    res.push_back(zero);
    store->idx = zero;
    store->value = value;
    store->dtype = arg->dtype;
    res.push_back(store);
    return res;
  }

  std::vector<std::shared_ptr<BaseExpr>> muls;

  for (int i = 0; i < store_idxs.size(); i++) {
    auto mul = std::make_shared<BinaryExpr>();
    mul->op = BinaryOpKind::Mul;
    mul->lhs = store_idxs[i];
    mul->rhs = tensor_strides[i];
    res.push_back(mul);
    muls.push_back(mul);
  }

  // now final expression summing all
  int x = muls.size();
  auto sum = binop_chain(muls, BinaryOpKind::Add);

  res.insert(res.end(), sum.begin(), sum.end());
  // now, do the store
  auto store = std::make_shared<StoreExpr>();
  store->ptr = arg;
  store->idx = sum.back();
  store->value = value;
  store->dtype = arg->dtype;
  res.push_back(store);
  return res;
}

} // namespace ir
class Compiled : public ADPrimitive {
public:
  std::string str() { return "Compiled"; }
  DEFINE_DISPATCH_CUDA
  ir::ir_t ir;
  std::map<int, std::vector<std::shared_ptr<ir::BaseExpr>>>
      tensor_idx_to_strides;
  void *cached_fn = nullptr;
};

} // namespace pg