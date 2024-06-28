#include "ir.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <nvrtc.h>
#include <sstream>
#include <string>
#include <unordered_map>

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
  } else if (is<Gt>(prim)) {
    return BinaryOpKind::Gt;
  } else if (is<Lt>(prim)) {
    return BinaryOpKind::Lt;
  } else if (is<Max>(prim)) {
    return BinaryOpKind::Max;
  } else if (is<Eq>(prim)) {
    return BinaryOpKind::Eq;
  } else if (is<Pow>(prim)) {
    return BinaryOpKind::Pow;

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
  } else if (is<Gt>(prim)) {
    return BinaryOpKind::Gt;
  } else if (is<Lt>(prim)) {
    return BinaryOpKind::Lt;
  } else if (is<Max>(prim)) {
    return BinaryOpKind::Max;
  } else if (is<Eq>(prim)) {
    return BinaryOpKind::Eq;
  } else if (is<Pow>(prim)) {
    return BinaryOpKind::Pow;

  } else {
    throw std::runtime_error("Unsupported binary operation");
  }
}

static UnaryOpKind op_to_unaryop_kind(ADPrimitive &prim) {
  if (is<Log>(prim)) {
    return UnaryOpKind::Log;
  } else if (is<Exp>(prim)) {
    return UnaryOpKind::Exp;
  } else {
    throw std::runtime_error("Unsupported unary operation");
  }
}

static TernaryOpKind op_to_ternaryop_kind(ADPrimitive &prim) {
  if (is<Where>(prim)) {
    return TernaryOpKind::Where;
  } else {
    throw std::runtime_error("Unsupported ternary operation");
  }
}

static UnaryOpKind op_to_unaryop_kind(std::shared_ptr<ADPrimitive> prim) {
  if (is<Log>(prim)) {
    return UnaryOpKind::Log;
  } else if (is<Exp>(prim)) {
    return UnaryOpKind::Exp;
  } else {
    throw std::runtime_error("Unsupported unary operation");
  }
}

static bool is_binary_op(ADPrimitive &prim) {
  return is<Add>(prim) || is<Sub>(prim) || is<Mul>(prim) || is<Div>(prim) ||
         is<Gt>(prim) || is<Lt>(prim) || is<Max>(prim) || is<Eq>(prim) ||
         is<Pow>(prim);
}

static bool is_binary_op(std::shared_ptr<ADPrimitive> prim) {
  return is<Add>(prim) || is<Sub>(prim) || is<Mul>(prim) || is<Div>(prim) ||
         is<Gt>(prim) || is<Lt>(prim) || is<Max>(prim) || is<Eq>(prim) ||
         is<Pow>(prim);
}

std::shared_ptr<BaseExpr>
graph_to_ir_inner(Tensor &out, std::vector<std::shared_ptr<BaseExpr>> &ir,
                  IrBuilderContext &ctx,
                  const std::vector<Tensor> &orig_inputs) {
  auto prim = out.ad_node().primitive();
  // first detect constants, then args, then binary ops, then unary ops
  if (is<Fill>(prim)) {
    auto fill = std::make_shared<ImmExpr>();
    fill->value = as<Fill>(prim)->value();
    fill->dtype = out.dtype();
    ir.push_back(fill);
    return fill;
  }

  auto is_in_orig_inputs = std::find_if(orig_inputs.begin(), orig_inputs.end(),
                                        [&out](const Tensor &t) {
                                          return t.id == out.id;
                                        }) != orig_inputs.end();

  if (is_in_orig_inputs) {
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
      // if stride size (load_idxs) is 0, then we load from [0]
      if (load_idxs.size() == 0) {
        auto load = std::make_shared<LoadExpr>();
        load->child = arg;
        auto zero = std::make_shared<ImmExpr>();
        zero->value = 0;
        ir.push_back(zero);
        load->idx = zero;
        load->dtype = arg->dtype;
        ir.push_back(load);
        return load;
      }

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

  // first render the input tensors
  std::vector<std::shared_ptr<BaseExpr>> inputs;
  for (auto &input : out.children()) {
    auto ir_ = graph_to_ir_inner(input, ir, ctx, orig_inputs);
    inputs.push_back(ir_);
  }
  // then render the current tensor, based on the inputs

  if (is_binary_op(prim)) {
    auto binop = std::make_shared<BinaryExpr>();
    binop->op = op_to_binop_kind(prim);
    binop->lhs = inputs[0];
    binop->rhs = inputs[1];
    ir.push_back(binop);
    return binop;
  }
  if (is<Log>(prim) || is<Exp>(prim)) {
    auto unop = std::make_shared<UnaryExpr>();
    unop->op = op_to_unaryop_kind(prim);
    unop->child = inputs[0];
    ir.push_back(unop);
    return unop;
  }
  if (is<Where>(prim)) {
    auto ternop = std::make_shared<TernaryExpr>();
    ternop->op = TernaryOpKind::Where;
    ternop->first = inputs[0];
    ternop->second = inputs[1];
    ternop->third = inputs[2];
    ir.push_back(ternop);
    return ternop;
  }
  throw std::runtime_error(
      "Bad schedule. Not an input and not a supported op: out: " + out.str());
}

void optim_ir_implace(ir_t &ir) {
  // if it is a pow with int, replace
  for (int i = 0; i < ir.size(); i++) {
    auto expr = ir[i];
    ir_t new_interm_ir = ir_t();
    if (is<BinaryExpr>(expr)) {
      auto binop = as<BinaryExpr>(expr);
      if (binop->op == BinaryOpKind::Pow) {
        if (is<ImmExpr>(binop->rhs)) {
          auto imm = as<ImmExpr>(binop->rhs);
          // if is int less than 10
          double val = imm->value;
          bool is_int = val == (int)val;

          if (is_int && val < 10) {
            // replace with a series of multiplications
            auto lhs = binop->lhs;
            auto res = lhs;
            for (int j = 1; j < val; j++) {
              auto new_binop = std::make_shared<BinaryExpr>();
              new_binop->op = BinaryOpKind::Mul;
              new_binop->lhs = res;
              new_binop->rhs = lhs;
              new_interm_ir.push_back(new_binop);
              res = new_binop;
            }

            // delete the old pow, add new_interm_ir
            ir.erase(ir.begin() + i);
            ir.insert(ir.begin() + i, new_interm_ir.begin(),
                      new_interm_ir.end());

            // find all references to the old pow and replace with the new res
            auto toreplace = expr;
            for (int j = 0; j < ir.size(); j++) {
              auto expr = ir[j];
              if (is<BinaryExpr>(expr)) {
                auto binop = as<BinaryExpr>(expr);
                if (binop->lhs == toreplace) {
                  binop->lhs = res;
                }
                if (binop->rhs == toreplace) {
                  binop->rhs = res;
                }
              }
              if (is<UnaryExpr>(expr)) {
                auto unop = as<UnaryExpr>(expr);
                if (unop->child == toreplace) {
                  unop->child = res;
                }
              }
              if (is<TernaryExpr>(expr)) {
                auto ternop = as<TernaryExpr>(expr);
                if (ternop->first == toreplace) {
                  ternop->first = res;
                }
                if (ternop->second == toreplace) {
                  ternop->second = res;
                }
                if (ternop->third == toreplace) {
                  ternop->third = res;
                }
              }
              if (is<LoadExpr>(expr)) {
                auto load = as<LoadExpr>(expr);
                if (load->idx == toreplace) {
                  load->idx = res;
                }
                if (load->child == toreplace) {
                  load->child = res;
                }
              }
              if (is<StoreExpr>(expr)) {
                auto store = as<StoreExpr>(expr);
                if (store->idx == toreplace) {
                  store->idx = res;
                }
                if (store->value == toreplace) {
                  store->value = res;
                }
              }

              if (is<IfStartExpr>(expr)) {
                auto if_start = as<IfStartExpr>(expr);
                if (if_start->cond == toreplace) {
                  if_start->cond = res;
                }
              }

              if (is<ReturnExpr>(expr)) {
                auto ret = as<ReturnExpr>(expr);
                if (ret->value == toreplace) {
                  ret->value = res;
                }
              }

              if (is<ForStartExpr>(expr)) {
                auto for_start = as<ForStartExpr>(expr);
                if (for_start->start == toreplace) {
                  for_start->start = res;
                }
                if (for_start->end == toreplace) {
                  for_start->end = res;
                }
                if (for_start->step == toreplace) {
                  for_start->step = res;
                }
              }

              if (is<IfEndExpr>(expr)) {
                auto if_end = as<IfEndExpr>(expr);
                if (if_end->if_start == toreplace) {
                  if_end->if_start = res;
                }
              }
            }
          }
        }
      }
    }
  }
}

std::tuple<std::vector<std::shared_ptr<BaseExpr>>, IrBuilderContext,
           std::vector<bool>>
graph_to_ir(Tensor &out, const std::vector<Tensor> &inputs) {
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
  global_idx->force_render = true;
  global_idx->name = "global_idx";

  ir.push_back(lhs->lhs);
  ir.push_back(lhs->rhs);
  ir.push_back(lhs);
  ir.push_back(rhs);
  ir.push_back(global_idx);

  int gidx_idx = ir.size() - 1;

  // add 'if (global_idx < numel) { return; }' to the ir

  auto numel = std::make_shared<ImmExpr>();
  numel->value = out.numel();
  ir.push_back(numel);

  auto cmp = std::make_shared<BinaryExpr>();
  cmp->op = BinaryOpKind::Lt;
  cmp->lhs = numel;
  cmp->rhs = global_idx;
  ir.push_back(cmp);

  auto if_expr = std::make_shared<IfStartExpr>();
  if_expr->cond = cmp;
  ir.push_back(if_expr);

  // if the condition is false, we must return
  auto ret = std::make_shared<ReturnExpr>();
  ir.push_back(ret);

  // end of the if
  auto if_end = std::make_shared<IfEndExpr>();
  ir.push_back(if_end);
  // no else

  // fill the ctx and ir with the input tensors
  int i = 0;
  int impidx = 0;
  int absi = 0;
  std::vector<bool> used_inputs(inputs.size(), false);
  for (auto &input : inputs) {
    if (is<Fill>(input.ad_node().primitive())) {
      absi++;
      // will be replaced by a constant
      continue;
    }

    used_inputs[absi] = true;
    absi++;

    auto arg = std::make_shared<ArgExpr>();
    arg->dtype = input.dtype();
    ir.push_back(arg);
    ctx.args.push_back(arg);
    ctx.tensor_id_to_ir_idx[input.id] = i;
    // fill the context for doing a load expression
    ContextForDoingALoadExpr arg_ctx;
    arg_ctx.is_contiguous = input.is_contiguous();
    auto cont = arg_ctx.is_contiguous;
    if (!cont) {
      // placeholder strides, we will fill them later
      strides_t strides = strides_t();

      for (int k = 0; k < input.ndim(); k++) {
        strides.push_back(input.strides()[k] / dtype_to_size(input.dtype()));
      }
      // first case, non contiguous
      // then we must create a local idx based from the global idx for each dim
      std::vector<std::shared_ptr<BaseExpr>> shapes_to_div =
          std::vector<std::shared_ptr<BaseExpr>>();
      for (int j = input.ndim() - 1; j >= 0; j--) {
        auto stride = std::make_shared<ImmExpr>();
        stride->value = strides[j];
        ir.push_back(stride);
        ctx.tensor_idx_to_strides[impidx].push_back(
            stride); // we will update this later
        arg_ctx.stride_exprs_idxs.push_back(ir.size() - 1);
        auto shape = std::make_shared<ImmExpr>();
        // shape is already inferred
        shape->value = input.shape()[j];
        ir.push_back(shape);
        arg_ctx.shape_exprs_idxs.push_back(ir.size() - 1);
        // now, the expression is expr = global_idx / (shapes_to_div_0 *
        // shapes_to_div_1 * ... * shapes_to_div_n) % shape
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

        // now local_idx = (global_idx / shapes_mul_accum) % shape
        auto local_idx = std::make_shared<BinaryExpr>();

        auto div = std::make_shared<BinaryExpr>();
        div->op = BinaryOpKind::Div;
        div->lhs = global_idx;
        div->rhs = mod_lhs;

        ir.push_back(div);

        local_idx->lhs = div;
        local_idx->rhs = shape;
        local_idx->op = BinaryOpKind::Mod;
        ir.push_back(local_idx);

        // force local_idx to render
        local_idx->name =
            "arg_" + std::to_string(impidx) + "_idx_" + std::to_string(j);
        local_idx->force_render = true;

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
    impidx++;
  }

  // same, but for the output tensor
  auto arg = std::make_shared<ArgExpr>();
  ir.push_back(arg);
  arg->dtype = out.dtype();
  ctx.args.push_back(arg);
  ctx.tensor_id_to_ir_idx[out.id] = i;
  // fill the context for doing a load expression
  ContextForDoingALoadExpr arg_ctx;
  // assume output is contiguous
  arg_ctx.is_contiguous = true;
  arg_ctx.load_idx_exprs_idxs.push_back(gidx_idx);
  ctx.arg_to_ctx[arg] = arg_ctx;
  i++;
  graph_to_ir_inner(out, ir, ctx, inputs);

  // now, at the end, add a store operation
  auto store = std::make_shared<StoreExpr>();
  store->ptr = ctx.args.back();
  store->value = ir.back();
  store->idx = ir[gidx_idx];
  ir.push_back(store);
  // optim_ir_implace(ir);
  return {ir, ctx, used_inputs};
}

static std::string binop_kind_to_str(BinaryOpKind op, std::string lhs,
                                     std::string rhs) {
  switch (op) {
  case BinaryOpKind::Add:
    return lhs + " + " + rhs;
  case BinaryOpKind::Sub:
    return lhs + " - " + rhs;
  case BinaryOpKind::Mul:
    return lhs + " * " + rhs;
  case BinaryOpKind::Div:
    return lhs + " / " + rhs;
  case BinaryOpKind::Gt:
    return lhs + " > " + rhs;
  case BinaryOpKind::Lt:
    return lhs + " < " + rhs;
  case BinaryOpKind::Max:
    return "max(" + lhs + ", " + rhs + ")";
  case BinaryOpKind::Mod:
    return lhs + " % " + rhs;
  case BinaryOpKind::Eq:
    return lhs + " == " + rhs;
  case BinaryOpKind::Pow: // todo - differentiate between int and float pow
    return "powf(" + lhs + ", " + rhs + ")";
  default:
    throw std::runtime_error("Unsupported binary operation: " +
                             std::to_string((int)op));
  }
}

static std::string unop_kind_to_str(UnaryOpKind op) {
  switch (op) {
  case UnaryOpKind::Log:
    return "log";
  case UnaryOpKind::Exp:
    return "exp";
  default:
    throw std::runtime_error("Unsupported unary operation");
  }
}

static std::string ternop_kind_to_str(TernaryOpKind op, std::string first,
                                      std::string second, std::string third) {
  switch (op) {
  case TernaryOpKind::Where:
    return first + " ? " + second + " : " + third;
  default:
    throw std::runtime_error("Unsupported ternary operation");
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
    // if already has a name, skip
    if (expr->name != "") {
      continue;
    }
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
  return "not implemented";
}

// THE PREVIOUJS ONE IS FOR VISUALIZATION
// THIS MUST OUTPUT A STRING THAT CAN BE COMPILED
std::string ir_to_cuda(std::vector<std::shared_ptr<BaseExpr>> &ir) {
  assign_names_to_ir(ir);
  std::string res = render_fn_header("", ir) + "\n";
  res = "__global__ void kernel_name(" + res + ")\n";
  res += "{\t\n";
  std::map<std::shared_ptr<BaseExpr>, std::string> r;
  std::map<std::string, bool> rendered; // used for example not to render block
                                        // dim twice
  NameDatabase nmdb;
  for (auto &expr : ir) {
    if (is<BinaryExpr>(expr)) {
      auto binop = as<BinaryExpr>(expr);
      /*res += get_dtype_cpp_str(binop->dtype) + " " + binop->name + " = " +
             binop->lhs->name + " " + binop_kind_to_str(binop->op) + " " +
             binop->rhs->name + ";\n";*/
      if (!binop->force_render) {
        r[expr] = "(" +
                  binop_kind_to_str(binop->op, r[binop->lhs], r[binop->rhs]) +
                  ")";
      } else {
        r[expr] = binop->name;
        res += get_dtype_cpp_str(binop->dtype) + " " + binop->name + " = " +
               binop_kind_to_str(binop->op, r[binop->lhs], r[binop->rhs]) +
               ";\n";
      }
    } else if (is<UnaryExpr>(expr)) {
      auto unop = as<UnaryExpr>(expr);
      /*res += get_dtype_cpp_str(unop->dtype) + " " + unop->name + " = " +
             unop_kind_to_str(unop->op) + "(" + unop->child->name + ");\n";*/
      if (!unop->force_render) {
        r[expr] = unop_kind_to_str(unop->op) + "(" + r[unop->child] + ")";
      } else {
        r[expr] = unop->name;
        res += get_dtype_cpp_str(unop->dtype) + " " + unop->name + " = " +
               unop_kind_to_str(unop->op) + "(" + r[unop->child] + ");\n";
      }
    } else if (is<ImmExpr>(expr)) {
      auto imm = as<ImmExpr>(expr);
      // switch type
      if (imm->dtype == DType::Int32) {
        r[expr] = std::to_string((int)imm->value);
      } else if (imm->dtype == DType::Float32) {
        // here we cannot use std::to_string because it will 'round' the float
        // to 6 digits
        std::ostringstream oss;
        oss << std::fixed
            << std::setprecision(std::numeric_limits<float>::max_digits10)
            << imm->value;
        r[expr] = oss.str();
      } else {
        PG_CHECK_RUNTIME(false, "Unsupported dtype");
      }
    } else if (is<ArgExpr>(expr)) {
      continue; // already rendered in the header
    } else if (is<BlockIdxExpr>(expr)) {
      auto block_idx = as<BlockIdxExpr>(expr);
      if (!rendered["bidx"]) {
        res += get_dtype_cpp_str(DType::Int32) + " " + "bidx" + " = " +
               "blockIdx.x" + ";\n";
        rendered["bidx"] = true;
      }
      r[expr] = "bidx";
    } else if (is<BlockDimExpr>(expr)) {
      auto block_dim = as<BlockDimExpr>(expr);
      if (!rendered["bdim"]) {
        res += get_dtype_cpp_str(DType::Int32) + " " + "bdim" + " = " +
               "blockDim.x" + ";\n";
        rendered["bdim"] = true;
      }
      r[expr] = "bdim";
    } else if (is<ThreadIdxExpr>(expr)) {
      auto thread_idx = as<ThreadIdxExpr>(expr);
      if (!rendered["tidx"]) {
        res += get_dtype_cpp_str(DType::Int32) + " " + "tidx" + " = " +
               "threadIdx.x" + ";\n";
        rendered["tidx"] = true;
      }
      r[expr] = "tidx";
    } else if (is<LoadExpr>(expr)) {
      auto load = as<LoadExpr>(expr);
      /*res += get_dtype_cpp_str(load->dtype) + " " + expr->name + " = " +
             load->child->name + "[" + load->idx->name + "];\n";*/
      auto x = nmdb.get_with_prefix("load");
      r[expr] = x;
      res += get_dtype_cpp_str(load->dtype) + " " + x + " = " +
             load->child->name + "[" + r[load->idx] + "]" + ";\n";
    } else if (is<StoreExpr>(expr)) {
      auto store = as<StoreExpr>(expr);
      res += store->ptr->name + "[" + r[store->idx] + "] = " + r[store->value] +
             ";\n";
    } else if (is<IfStartExpr>(expr)) {
      auto if_start = as<IfStartExpr>(expr);
      res += "if (" + r[if_start->cond] + ") {\n";
    } else if (is<IfEndExpr>(expr)) {
      res += "}\n";
    } else if (is<ReturnExpr>(expr)) {
      res += "return;\n";
    } else if (is<TernaryExpr>(expr)) {
      auto ternop = as<TernaryExpr>(expr);
      r[expr] = "(" +
                ternop_kind_to_str(ternop->op, r[ternop->first],
                                   r[ternop->second], r[ternop->third]) +
                ")";
    } else {
      PG_CHECK_RUNTIME(false, "Unsupported expression: " + expr->expr_str());
    }
  }
  res += "}\n";
  return res;
}
} // namespace ir

void Compiled::dispatch_cuda(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  // first, we need to gather ir
  using namespace ir;

  auto out = outputs[0];

  if (this->cached_fn == nullptr) {

    // firsts
    std::string ker = ir_to_cuda(this->ir);
    nvrtcProgram prog;
    // apend extern C
    std::string file = "extern \"C\" {\n" + ker + "\n}";
    nvrtcCreateProgram(&prog, file.c_str(), nullptr, 0, nullptr, nullptr);

    if (std::getenv("PG_KERNEL_DB") != nullptr) {
      std::cout << "file: " << file << std::endl;
    }
    const char *opts[] = {"--use_fast_math"};
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);

    // Check for compilation errors
    if (compileResult != NVRTC_SUCCESS) {
      size_t logSize;
      nvrtcGetProgramLogSize(prog, &logSize);
      char *log = new char[logSize];
      nvrtcGetProgramLog(prog, log);
      nvrtcDestroyProgram(&prog);
      throw std::runtime_error("NVRTC compilation failed: " + std::string(log));
    }

    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char *ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    CUmodule cuModule;
    CUfunction cuFunction;
    CUcontext cuContext;
    CUresult R1 = cuModuleLoadData(&cuModule, ptx);
    PG_CHECK_RUNTIME(R1 == CUDA_SUCCESS,
                     "Failed to load data: got " + std::to_string(R1));
    std::string kernel_name = "kernel_name";
    CUresult R =
        cuModuleGetFunction(&cuFunction, cuModule, kernel_name.c_str());
    PG_CHECK_RUNTIME(R == CUDA_SUCCESS, "Failed to get function: got " +
                                            std::to_string(R) + " for kernel " +
                                            kernel_name);

    PG_CHECK_RUNTIME(cuFunction != nullptr, "Failed to get function");
    // Store the function pointer in a void*
    void *function_ptr = reinterpret_cast<void *>(cuFunction);
    PG_CHECK_RUNTIME(function_ptr != nullptr, "Failed to get function pointer");
    // Clean up
    nvrtcDestroyProgram(&prog);
    delete[] ptx;
    this->cached_fn = function_ptr;
  }

  outputs[0].view_ptr()->allocate();
  // Prepare kernel arguments
  // Prepare grid and block dimensions
  dim3 threads_per_block(128, 1, 1);
  size_t num_elements = outputs[0].numel();
  dim3 blocks_per_grid(
      (num_elements + threads_per_block.x - 1) / threads_per_block.x, 1, 1);
  std::vector<void *> kernel_args;
  for (const auto &input : inputs) {
    void *in_data = input.get_base_ptr();
    kernel_args.push_back(in_data);
  }
  void *out_data = outputs[0].get_base_ptr();
  kernel_args.push_back(out_data);

  // Convert to array of pointers
  std::vector<void *> kernel_args_ptrs;
  for (auto &arg : kernel_args) {
    kernel_args_ptrs.push_back(&arg);
  }

  // Launch the kernel
  // create stream to launch kernel
  // first check if function is valid
  CUresult launch_result = cuLaunchKernel(
      (CUfunction)this->cached_fn, blocks_per_grid.x, blocks_per_grid.y,
      blocks_per_grid.z, threads_per_block.x, threads_per_block.y,
      threads_per_block.z, 0, NULL, kernel_args_ptrs.data(), NULL);

  if (launch_result != CUDA_SUCCESS) {
    const char *error_string;
    cuGetErrorString(launch_result, &error_string);
    PG_CHECK_RUNTIME(
        false, "Error launching kernel: " + std::string(error_string) +
                   " "
                   "for kernel " +
                   std::to_string(outputs[0].id) +
                   " with args: " + vec_to_string(kernel_args_ptrs) +
                   "\n and fn_ptr: " + std::to_string((size_t)this->cached_fn));
  }

  // Synchronize to ensure kernel execution is complete
  cudaDeviceSynchronize();
  // check error again
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    PG_CHECK_RUNTIME(false,
                     "cuda error: " + std::string(cudaGetErrorString(error)));
  }
}

} // namespace pg