#include "ir.hpp"

namespace pg {
namespace ir {
static BinaryOpKind op_to_binop_kind(ADPrimitive &prim) {
  if (is<Add>(prim)) {
    return BinaryOpKind::Add;
  } else if (is<Sub>(prim)) {
    return BinaryOpKind::Sub;
  } else if (is<Mul>(prim)) {
    return BinaryOpKind::Mul;
  } else if (is<Div>(prim)) {
    return BinaryOpKind::Div;
  } else {
    throw std::runtime_error("Unsupported binary operation");
  }
}

static BinaryOpKind op_to_binop_kind(std::shared_ptr<ADPrimitive> prim) {
  if (is<Add>(prim)) {
    return BinaryOpKind::Add;
  } else if (is<Sub>(prim)) {
    return BinaryOpKind::Sub;
  } else if (is<Mul>(prim)) {
    return BinaryOpKind::Mul;
  } else if (is<Div>(prim)) {
    return BinaryOpKind::Div;
  } else {
    throw std::runtime_error("Unsupported binary operation");
  }
}

static bool is_binary_op(ADPrimitive &prim) {
  return is<Add>(prim) || is<Sub>(prim) || is<Mul>(prim) || is<Div>(prim);
}

static bool is_binary_op(std::shared_ptr<ADPrimitive> prim) {
  return is<Add>(prim) || is<Sub>(prim) || is<Mul>(prim) || is<Div>(prim);
}

class ContextForDoingALoadExpr {
public:
  bool is_contiguous;
  std::vector<int> stride_exprs_idxs;
  std::vector<int> shape_exprs_idxs;
  std::vector<int> load_idx_exprs_idxs;
};

class IrBuilderContext {
public:
  std::map<std::shared_ptr<ArgExpr>, ContextForDoingALoadExpr> arg_to_ctx;
  std::vector<std::shared_ptr<ArgExpr>> args;
  // tensor id -> ir expr idx
  std::map<int, int> tensor_id_to_ir_idx;
};

std::shared_ptr<BaseExpr>
graph_to_ir_inner(Tensor &out, std::vector<std::shared_ptr<BaseExpr>> &ir,
                  IrBuilderContext &ctx) {
  // first render the input tensors
  std::vector<std::shared_ptr<BaseExpr>> inputs;
  for (auto &input : out.children()) {
    auto ir_ = graph_to_ir_inner(input, ir, ctx);
    inputs.push_back(ir_);
  }
  // then render the current tensor, based on the inputs
  auto prim = out.ad_node().primitive();
  if (is_binary_op(prim)) {
    auto binop = std::make_shared<BinaryExpr>();
    binop->op = op_to_binop_kind(prim);
    binop->lhs = inputs[0];
    binop->rhs = inputs[1];
    ir.push_back(binop);
    return binop;
  }
  if (is<Fill>(prim)) {
    auto fill = std::make_shared<ImmExpr>();
    fill->value = as<Fill>(prim)->value();
    ir.push_back(fill);
    return fill;
  }

  if (is<JitBoundary>(prim)) {
    // these are the args to the kernel
    // we need to do a load expression
    // first, get the arg expr
    auto arg_idx = ctx.tensor_id_to_ir_idx.at(out.id);
    auto arg = ctx.args[arg_idx];
    auto arg_ctx = ctx.arg_to_ctx.at(arg);
    auto load_idxs = arg_ctx.load_idx_exprs_idxs;
    if (arg_ctx.is_contiguous) {
      // if contiguous, we can simply load from the global idx
      auto load = std::make_shared<LoadExpr>();
      load->child = arg;
      load->idx = ir[load_idxs[0]];
      load->dtype = arg->dtype;
      ir.push_back(load);
      return load;
    } else {
      // LOGIC IS HERE
      /*std::string render_idxs(std::vector<long long> &rendered_ids) override {
      // if our id is already rendered, we don't need to render it again
      if (std::find(rendered_ids.begin(), rendered_ids.end(), id) !=
          rendered_ids.end()) {
      return "";
      }
      // add our id to the rendered ids
      rendered_ids.push_back(id);
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
  }*/
      // EACH IDX IS A DIMENSION, AND IS STORED AT load_idxs
      // so we need to do
      // x = load_from[stride_0 * idx_0 + stride_1 * idx_1 + ... + stride_n *
      // idx_n]

      // first, calculate the idx
      std::vector<std::shared_ptr<BaseExpr>> muls;

      for (int i = 0; i < load_idxs.size(); i++) {
        auto mul = std::make_shared<BinaryExpr>();
        mul->op = BinaryOpKind::Mul;
        mul->lhs = ir[load_idxs[i]];
        mul->rhs = ir[arg_ctx.stride_exprs_idxs[i]];
        ir.push_back(mul);
        muls.push_back(mul);
      }

      // now final expression summing all
      int x = muls.size();
      if (x == 1) {
        // no need to sum
        auto load = std::make_shared<LoadExpr>();
        load->child = arg;
        load->idx = muls[0];
        load->dtype = arg->dtype;
        ir.push_back(load);
        return load;
      }
      auto sum = std::make_shared<BinaryExpr>();
      sum->op = BinaryOpKind::Add;
      sum->lhs = muls[0];
      sum->rhs = muls[1];
      ir.push_back(sum);
      for (int i = 2; i < x; i++) {
        auto new_sum = std::make_shared<BinaryExpr>();
        new_sum->op = BinaryOpKind::Add;
        new_sum->lhs = sum;
        new_sum->rhs = muls[i];
        ir.push_back(new_sum);
        sum = new_sum;
      }

      // now, do the load
      auto load = std::make_shared<LoadExpr>();
      load->child = arg;
      load->idx = sum;
      load->dtype = arg->dtype;
      ir.push_back(load);
      return load;
    }
  }
}

std::vector<std::shared_ptr<BaseExpr>>
graph_to_ir(Tensor &out, std::vector<Tensor> &inputs) {
  // the result will be a linear IR
  std::vector<std::shared_ptr<BaseExpr>> ir;
  IrBuilderContext ctx;

  // rn only works for cuda
  // declare global idx as a (blockIdx * blockDim + threadIdx)
  auto lhs = std::make_shared<BinaryExpr>();
  lhs->op = BinaryOpKind::Mul;
  lhs->lhs = std::make_shared<BlockIdxExpr>();
  lhs->rhs = std::make_shared<BlockDimExpr>();

  auto rhs = std::make_shared<ThreadIdxExpr>();

  auto global_idx = std::make_shared<BinaryExpr>();
  global_idx->op = BinaryOpKind::Add;
  global_idx->lhs = lhs;
  global_idx->rhs = rhs;

  ir.push_back(lhs->lhs);
  ir.push_back(lhs->rhs);
  ir.push_back(lhs);
  ir.push_back(rhs);
  ir.push_back(global_idx);

  int gidx_idx = ir.size() - 1;

  // fill the ctx and ir with the input tensors
  int i = 0;
  for (auto &input : inputs) {
    auto arg = std::make_shared<ArgExpr>();
    ir.push_back(arg);
    ctx.args.push_back(arg);
    ctx.tensor_id_to_ir_idx[input.id] = i;
    // fill the context for doing a load expression
    ContextForDoingALoadExpr arg_ctx;
    bool cont = false;
    arg_ctx.is_contiguous = cont;
    if (!cont) {
      // placeholder strides, we will fill them later
      strides_t strides = strides_t();
      // fill with 1..n
      for (int k = 1; k <= input.ndim(); k++) {
        strides.push_back(k);
      }
      // first case, non contiguous
      // then we must create a local idx based from the global idx for each dim
      std::vector<std::shared_ptr<BaseExpr>> shapes_to_div =
          std::vector<std::shared_ptr<BaseExpr>>();
      for (int j = 0; j < input.ndim(); j++) {
        auto stride = std::make_shared<ImmExpr>();
        stride->value = strides[j];
        ir.push_back(stride);
        arg_ctx.stride_exprs_idxs.push_back(ir.size() - 1);
        auto shape = std::make_shared<ImmExpr>();
        // shape is already inferred
        shape->value = input.shape()[j];
        ir.push_back(shape);
        arg_ctx.shape_exprs_idxs.push_back(ir.size() - 1);
        auto local_idx = std::make_shared<BinaryExpr>();

        // now, the expression is expr = global_idx / (shapes_to_div_0 *
        // shapes_to_div_1 * ... * shapes_to_div_n) % shape
        local_idx->op = BinaryOpKind::Div;
        std::shared_ptr<BaseExpr> mod_lhs;
        if (shapes_to_div.size() == 0) {
          auto shapes_mul_accum = std::make_shared<BinaryExpr>();
          shapes_mul_accum->op = BinaryOpKind::Mul;
          auto one1 = std::make_shared<ImmExpr>();
          one1->value = 1;
          auto one2 = std::make_shared<ImmExpr>();
          one2->value = 1;
          ir.push_back(one1);
          ir.push_back(one2);
          shapes_mul_accum->lhs = one1;
          shapes_mul_accum->rhs = one2;
          ir.push_back(shapes_mul_accum);
          mod_lhs = shapes_mul_accum;
        } else {
          mod_lhs = shapes_to_div[0];
          for (int k = 1; k < shapes_to_div.size(); k++) {
            auto new_mul = std::make_shared<BinaryExpr>();
            new_mul->op = BinaryOpKind::Mul;
            new_mul->lhs = mod_lhs;
            new_mul->rhs = shapes_to_div[k];
            ir.push_back(new_mul);
            mod_lhs = new_mul;
          }
        }

        // now local_idx = global_idx / shapes_mul_accum % shape
        local_idx->lhs = global_idx;
        auto rhs = std::make_shared<BinaryExpr>();
        rhs->op = BinaryOpKind::Mod;
        rhs->lhs = mod_lhs;
        rhs->rhs = shape;
        ir.push_back(rhs);
        local_idx->rhs = rhs;

        ir.push_back(local_idx);
        arg_ctx.load_idx_exprs_idxs.push_back(ir.size() - 1);
        shapes_to_div.push_back(shape);
      }
    } else {
      // second case
      // contigous, then we can simply load from the global idx
      arg_ctx.load_idx_exprs_idxs.push_back(gidx_idx);
    }
    ctx.arg_to_ctx[arg] = arg_ctx;
    i++;
  }

  // same, but for the output tensor
  auto arg = std::make_shared<ArgExpr>();
  ir.push_back(arg);
  ctx.args.push_back(arg);
  ctx.tensor_id_to_ir_idx[out.id] = i;
  // fill the context for doing a load expression
  ContextForDoingALoadExpr arg_ctx;
  // assume output is contiguous
  arg_ctx.is_contiguous = true;
  arg_ctx.load_idx_exprs_idxs.push_back(gidx_idx);
  ctx.arg_to_ctx[arg] = arg_ctx;
  i++;

  graph_to_ir_inner(out, ir, ctx);

  // now, at the end, add a store operation
  auto store = std::make_shared<StoreExpr>();
  store->ptr = ctx.args.back();
  store->value = ir.back();
  store->idx = ir[gidx_idx];
  ir.push_back(store);
  return ir;
}

static std::string binop_kind_to_str(BinaryOpKind op) {
  switch (op) {
  case BinaryOpKind::Add:
    return "+";
  case BinaryOpKind::Sub:
    return "-";
  case BinaryOpKind::Mul:
    return "*";
  case BinaryOpKind::Div:
    return "/";
  case BinaryOpKind::Gt:
    return ">";
  case BinaryOpKind::Lt:
    return "<";
  case BinaryOpKind::Mod:
    return "%";
  default:
    throw std::runtime_error("Unsupported binary operation");
  }
}

std::string get_dtype_cpp_str(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return "float";
  case DType::Int32:
    return "int";
  default:
    throw std::runtime_error("Unsupported dtype");
  }
}

void assign_names_to_ir(std::vector<std::shared_ptr<BaseExpr>> &ir) {
  NameDatabase name_db;
  for (auto &expr : ir) {
    auto n = name_db.get_name(expr);
    expr->name = n;
  }
}

std::string render_fn_header(std::string fn_name,
                             std::vector<std::shared_ptr<BaseExpr>> &ir) {
  // get args
  std::vector<std::shared_ptr<ArgExpr>> args;
  for (auto &expr : ir) {
    if (is<ArgExpr>(expr)) {
      args.push_back(as<ArgExpr>(expr));
    }
  }
  std::string res = "";
  for (int i = 0; i < args.size(); i++) {
    auto arg = args[i];
    res += get_dtype_cpp_str(arg->dtype) + " *" + arg->name;
    if (i != args.size() - 1) {
      res += ", ";
    }
  }

  return res;
}

std::string ir_to_string(std::vector<std::shared_ptr<BaseExpr>> &ir) {
  assign_names_to_ir(ir);
  std::string res = render_fn_header("kernel_name", ir) + "\n";
  res += "{\t\n";
  for (auto &expr : ir) {
    if (is<BinaryExpr>(expr)) {
      auto binop = as<BinaryExpr>(expr);
      res += binop->name + " = " + binop->lhs->name + " " +
             binop_kind_to_str(binop->op) + " " + binop->rhs->name + ";\n";
    } else if (is<ImmExpr>(expr)) {
      auto imm = as<ImmExpr>(expr);
      res += imm->name + " = " + std::to_string(imm->value) + ";\n";
    } else if (is<ArgExpr>(expr)) {
      continue; // already rendered in the header
    } else if (is<BlockIdxExpr>(expr)) {
      auto block_idx = as<BlockIdxExpr>(expr);
      res += block_idx->name + " = " + "blockIdx.x" + ";\n";
    } else if (is<BlockDimExpr>(expr)) {
      auto block_dim = as<BlockDimExpr>(expr);
      res += block_dim->name + " = " + "blockDim.x" + ";\n";
    } else if (is<ThreadIdxExpr>(expr)) {
      auto thread_idx = as<ThreadIdxExpr>(expr);
      res += thread_idx->name + " = " + "threadIdx.x" + ";\n";
    } else if (is<LoadExpr>(expr)) {
      auto load = as<LoadExpr>(expr);
      res += expr->name + " = " + load->child->name + "[" + load->idx->name +
             "];\n";
    } else if (is<StoreExpr>(expr)) {
      auto store = as<StoreExpr>(expr);
      res += store->ptr->name + "[" + store->idx->name +
             "] = " + store->value->name + ";\n";
    } else {
      PG_CHECK_RUNTIME(false, "Unsupported expression: " + expr->expr_str());
    }
  }
  res += "}\n";
  return res;
}

// THE PREVIOUJS ONE IS FOR VISUALIZATION
// THIS MUST OUTPUT A STRING THAT CAN BE COMPILED
std::string ir_to_cuda(std::vector<std::shared_ptr<BaseExpr>> &ir) {
  assign_names_to_ir(ir);
  std::string res = render_fn_header("", ir) + "\n";
  res = "__global__ kernel_name(" + res + ")\n";
  res += "{\t\n";
  for (auto &expr : ir) {
    if (is<BinaryExpr>(expr)) {
      auto binop = as<BinaryExpr>(expr);
      res += get_dtype_cpp_str(binop->dtype) + " " + binop->name + " = " +
             binop->lhs->name + " " + binop_kind_to_str(binop->op) + " " +
             binop->rhs->name + ";\n";
    } else if (is<ImmExpr>(expr)) {
      auto imm = as<ImmExpr>(expr);
      res += get_dtype_cpp_str(imm->dtype) + " " + imm->name + " = " +
             std::to_string(imm->value) + ";\n";
    } else if (is<ArgExpr>(expr)) {
      continue; // already rendered in the header
    } else if (is<BlockIdxExpr>(expr)) {
      auto block_idx = as<BlockIdxExpr>(expr);
      res += get_dtype_cpp_str(DType::Int32) + " " + block_idx->name + " = " +
             "blockIdx.x" + ";\n";
    } else if (is<BlockDimExpr>(expr)) {
      auto block_dim = as<BlockDimExpr>(expr);
      res += get_dtype_cpp_str(DType::Int32) + " " + block_dim->name + " = " +
             "blockDim.x" + ";\n";
    } else if (is<ThreadIdxExpr>(expr)) {
      auto thread_idx = as<ThreadIdxExpr>(expr);
      res += get_dtype_cpp_str(DType::Int32) + " " + thread_idx->name + " = " +
             "threadIdx.x" + ";\n";
    } else if (is<LoadExpr>(expr)) {
      auto load = as<LoadExpr>(expr);
      res += get_dtype_cpp_str(load->dtype) + " " + expr->name + " = " +
             load->child->name + "[" + load->idx->name + "];\n";
    } else if (is<StoreExpr>(expr)) {
      auto store = as<StoreExpr>(expr);
      res += store->ptr->name + "[" + store->idx->name +
             "] = " + store->value->name + ";\n";
    } else {
      PG_CHECK_RUNTIME(false, "Unsupported expression: " + expr->expr_str());
    }
  }
  res += "}\n";
  return res;
}
} // namespace ir
} // namespace pg