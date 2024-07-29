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

static bool is_binary_op(std::shared_ptr<ADPrimitive> prim) {
  return is<Add>(prim) || is<Sub>(prim) || is<Mul>(prim) || is<Div>(prim) ||
         is<Gt>(prim) || is<Lt>(prim) || is<Max>(prim) || is<Eq>(prim) ||
         is<Pow>(prim);
}

static bool is_unary_op(std::shared_ptr<ADPrimitive> prim) {
  return is<Log>(prim) || is<Exp>(prim);
}

static bool is_ternary_op(std::shared_ptr<ADPrimitive> prim) {
  return is<Where>(prim);
}

static bool is_reduce_op(std::shared_ptr<ADPrimitive> prim) {
  return is<Sum>(prim) || is<Mean>(prim) || is<MaxReduce>(prim);
}

// we can treat f(Fill(value)) as a constant as long as f is a reshape,
// transpose or other movement op this function gets a tensor and return null if
// it cannot reach a Fill node, or the value of the Fill node if it can
static std::shared_ptr<ImmExpr> get_fill_value(const Tensor &t) {
  if (is<Fill>(t.ad_node()->primitive())) {
    auto fill = as<Fill>(t.ad_node()->primitive());
    auto imm = std::make_shared<ImmExpr>();
    imm->value = fill->value();
    imm->dtype = t.dtype();
    return imm;
  }
  if (!is<Reshape>(t.ad_node()->primitive()) &&
      !is<Permute>(t.ad_node()->primitive()) &&
      !is<BroadcastTo>(t.ad_node()->primitive())) {
    return nullptr;
  }
  for (auto &child : t.ad_node()->children()) {
    auto res = get_fill_value(child);
    if (res != nullptr) {
      return res;
    }
  }
  return nullptr;
}

std::shared_ptr<BaseExpr>
graph_to_ir_inner(Tensor &out, std::vector<Tensor> &marked_as_out,
                  std::vector<std::shared_ptr<BaseExpr>> &ir,
                  IrBuilderContext &ctx,
                  const std::vector<Tensor> &orig_inputs) {
  auto prim = out.ad_node()->primitive();
  // first detect constants, then args, then binary ops, then unary ops
  if (is<Fill>(prim)) {
    auto fill = std::make_shared<ImmExpr>();
    fill->value = as<Fill>(prim)->value();
    fill->dtype = out.dtype();
    ir.push_back(fill);
    return fill;
  }
  // we can treat f(Fill(value)) as a constant as long as f is a reshape,
  // transpose or other movement op
  auto fill_value = get_fill_value(out);
  if (fill_value != nullptr) {
    ir.push_back(fill_value);
    return fill_value;
  }

  auto is_in_orig_inputs = std::find_if(orig_inputs.begin(), orig_inputs.end(),
                                        [&out](const Tensor &t) {
                                          return t.id == out.id;
                                        }) != orig_inputs.end();

  if (is_in_orig_inputs) {
    // these are the args to the kernel
    // we need to do a load expression
    // first, get the arg expr
    // print all the contents of tid_to_arg
    auto arg = ctx.tid_to_arg.at(out.id);
    auto strides = ctx.arg_to_strides.at(arg);
    auto idxs_to_load = ctx.arg_to_idxs_to_load.at(arg);
    auto ir_load = render_load_idxs_for_expr(idxs_to_load, strides, arg);
    // irload is a vector
    ir.insert(ir.end(), ir_load.begin(), ir_load.end());
    return ir_load.back();
  }

  // first recursively the input tensors
  std::vector<std::shared_ptr<BaseExpr>> inputs;
  for (auto &input : out.children()) {
    auto ir_ = graph_to_ir_inner(input, marked_as_out, ir, ctx, orig_inputs);
    inputs.push_back(ir_);
  }

  // then render the current tensor, based on the inputs
  if (is_binary_op(prim)) {
    auto binop = std::make_shared<BinaryExpr>();
    binop->op = op_to_binop_kind(prim);
    binop->lhs = inputs[0];
    binop->rhs = inputs[1];
    ir.push_back(binop);
    // if it is marked as output, we need to store it
    if (std::find_if(marked_as_out.begin(), marked_as_out.end(),
                     [&out](const Tensor &t) { return t.id == out.id; }) !=
        marked_as_out.end()) {
      auto arg = ctx.tid_to_arg.at(out.id);
      auto strides = ctx.arg_to_strides.at(arg);
      auto store_ir = render_store_idxs_for_expr(
          ctx.arg_to_idxs_to_load.at(arg), strides, arg, binop);
      ir.insert(ir.end(), store_ir.begin(), store_ir.end());
    }
    return binop;
  }
  if (is_unary_op(prim)) {
    auto unop = std::make_shared<UnaryExpr>();
    unop->op = op_to_unaryop_kind(prim);
    unop->child = inputs[0];
    ir.push_back(unop);
    if (std::find_if(marked_as_out.begin(), marked_as_out.end(),
                     [&out](const Tensor &t) { return t.id == out.id; }) !=
        marked_as_out.end()) {
      auto arg = ctx.tid_to_arg.at(out.id);
      auto strides = ctx.arg_to_strides.at(arg);
      auto store_ir = render_store_idxs_for_expr(
          ctx.arg_to_idxs_to_load.at(arg), strides, arg, unop);
      ir.insert(ir.end(), store_ir.begin(), store_ir.end());
    }
    return unop;
  }
  if (is_ternary_op(prim)) {
    auto ternop = std::make_shared<TernaryExpr>();
    ternop->op = TernaryOpKind::Where;
    ternop->first = inputs[0];
    ternop->second = inputs[1];
    ternop->third = inputs[2];
    ir.push_back(ternop);
    if (std::find_if(marked_as_out.begin(), marked_as_out.end(),
                     [&out](const Tensor &t) { return t.id == out.id; }) !=
        marked_as_out.end()) {
      auto arg = ctx.tid_to_arg.at(out.id);
      auto strides = ctx.arg_to_strides.at(arg);
      auto store_ir = render_store_idxs_for_expr(
          ctx.arg_to_idxs_to_load.at(arg), strides, arg, ternop);
      ir.insert(ir.end(), store_ir.begin(), store_ir.end());
    }
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

static ir_t render_return_guard(int max_val, std::shared_ptr<BaseExpr> idx) {
  ir_t res;
  auto imm = std::make_shared<ImmExpr>();
  imm->value = max_val;
  auto cmp = std::make_shared<BinaryExpr>();
  cmp->op = BinaryOpKind::Lt;
  cmp->lhs = imm;
  cmp->rhs = idx;
  res.push_back(imm);
  res.push_back(cmp);
  auto if_expr = std::make_shared<IfStartExpr>();
  if_expr->cond = cmp;
  res.push_back(if_expr);
  auto ret = std::make_shared<ReturnExpr>();
  res.push_back(ret);
  auto if_end = std::make_shared<IfEndExpr>();
  res.push_back(if_end);
  return res;
}

using l = std::function<bool(int)>;
l default_choose_which_idxs_to_load = [](int i) { return true; };

static std::pair<ir_t, ir_t> render_local_idxs(
    ir_item_t gidx, ir_t shapes, int input_idx,
    l choose_which_idxs_to_load = default_choose_which_idxs_to_load) {
  ir_t res;
  ir_t only_loads = ir_t();
  only_loads.resize(shapes.size());
  std::vector<std::shared_ptr<BaseExpr>> shapes_to_div =
      std::vector<std::shared_ptr<BaseExpr>>();
  for (int j = shapes.size() - 1; j >= 0; j--) {
    if (!choose_which_idxs_to_load(j)) {
      continue;
    }
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
      res.push_back(one1);
      res.push_back(one2);
      shapes_mul_accum->lhs = one1;
      shapes_mul_accum->rhs = one2;
      res.push_back(shapes_mul_accum);
      mod_lhs = shapes_mul_accum;
    } else {
      mod_lhs = shapes_to_div[0];
      for (int k = 1; k < shapes_to_div.size(); k++) {
        auto new_mul = std::make_shared<BinaryExpr>();
        new_mul->op = BinaryOpKind::Mul;
        new_mul->lhs = mod_lhs;
        new_mul->rhs = shapes_to_div[k];
        res.push_back(new_mul);
        mod_lhs = new_mul;
      }
    }

    // now local_idx = (global_idx / shapes_mul_accum) % shape
    auto local_idx = std::make_shared<BinaryExpr>();

    auto div = std::make_shared<BinaryExpr>();
    div->op = BinaryOpKind::Div;
    div->lhs = gidx;
    div->rhs = mod_lhs;

    res.push_back(div);

    local_idx->lhs = div;
    local_idx->rhs = shapes[j];
    local_idx->op = BinaryOpKind::Mod;
    res.push_back(local_idx);

    // force local_idx to render
    local_idx->name =
        "arg_" + std::to_string(input_idx) + "_idx_" + std::to_string(j);
    local_idx->force_render = true;

    shapes_to_div.push_back(shapes[j]);
    only_loads[j] = local_idx;
  }

  return {res, only_loads};
}

std::tuple<std::vector<std::shared_ptr<BaseExpr>>, IrBuilderContext,
           std::vector<bool>>
graph_to_ir_reduce(Tensor &out, const std::vector<Tensor> &inputs) {
  PG_CHECK_RUNTIME(is_reduce_op(out.ad_node()->primitive()),
                   "graph_to_ir_reduce can only be called with a reduce op");

  std::shared_ptr<Reduce> reduce =
      std::dynamic_pointer_cast<Reduce>(out.ad_node()->primitive());
  axes_t axes = reduce->axes();
  for (int xx = 0; xx < axes.size(); xx++) {
    axes[xx] = axes[xx] < 0 ? out.ad_node()->children()[0].ndim() + axes[xx]
                            : axes[xx];
  }
  auto is_reduced = [&axes](int i) {
    return std::find(axes.begin(), axes.end(), i) != axes.end();
  };

  int total_out_elems = out.numel();
  int total_reduced_elems = reduce->total_reduce_numel();
  // out = reduce(fn(inputs), axis=...
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

  // we will render reduce like:
  /*
  void kernel(args...) {
    int gidx = ... // represents an element in the output tensor
    int out_local_idx_1 = ...
    int out_local_idx_0 = ...
    int in_local_idx_1 = ...
    ...
    for (int i = 0; i < total_reduce_numel; i++) { // in this case we reduce
  over axis 0 int in_local_idx_0 = ...
      ...
      // out is a reduction of in, so in idxs depends on the gidx and the 'i'
  idx (the reduction) float val = fn(in[in_local_idx_0, in_local_idx_1, ...]);
      out[out_local_idx_0, out_local_idx_1, ...] = val;
    }
  }
  */

  // add 'if (global_idx < numel) { return; }' to the ir
  auto return_guard = render_return_guard(out.numel(), global_idx);
  ir.insert(ir.end(), return_guard.begin(), return_guard.end());

  // fill the ctx and ir with the input tensors
  int i = 0;
  std::vector<bool> used_inputs(inputs.size(), false);
  for (auto &input : inputs) {
    if (is<Fill>(input.ad_node()->primitive())) {
      i++;
      // will be replaced by a constant
      continue;
    }
    used_inputs[i] = true;
    auto arg = std::make_shared<ArgExpr>();
    arg->dtype = input.dtype();
    ir.push_back(arg);
    ctx.tid_to_arg[input.id] = arg;
    // add to ctx.tid_to_shape and tid_to_strides
    ctx.arg_to_shape[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_strides[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_idxs_to_load[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_idxs_to_load[arg].resize(input.ndim());
    for (int j = 0; j < input.ndim(); j++) {
      auto shape = std::make_shared<ImmExpr>();
      shape->value = input.shape()[j];
      ir.push_back(shape);
      ctx.arg_to_shape.at(arg).push_back(shape);
      auto stride = std::make_shared<ImmExpr>();
      stride->value = input.strides()[j] /
                      dtype_to_size(input.dtype()); // stride is in bytes
      ir.push_back(stride);
      ctx.arg_to_strides.at(arg).push_back(stride);
    }
    auto [local_idxs, only_loads] =
        render_local_idxs(global_idx, ctx.arg_to_shape.at(arg), i,
                          [=](int i) { return !is_reduced(i); });
    ir.insert(ir.end(), local_idxs.begin(), local_idxs.end());
    ctx.arg_to_idxs_to_load[arg] = only_loads;
    i++;
  }

  // now same but for the output tensor
  auto arg = std::make_shared<ArgExpr>();
  arg->dtype = out.dtype();
  ir.push_back(arg);
  ctx.tid_to_arg[out.id] = arg;
  ctx.arg_to_shape[arg] = std::vector<std::shared_ptr<BaseExpr>>();
  ctx.arg_to_strides[arg] = std::vector<std::shared_ptr<BaseExpr>>();
  ctx.arg_to_idxs_to_load[arg] = std::vector<std::shared_ptr<BaseExpr>>();
  ctx.arg_to_idxs_to_load[arg].resize(out.ndim());
  for (int j = 0; j < out.ndim(); j++) {

    auto shape = std::make_shared<ImmExpr>();
    shape->value = out.shape()[j];
    ir.push_back(shape);
    ctx.arg_to_shape.at(arg).push_back(shape);
    auto stride = std::make_shared<ImmExpr>();
    stride->value =
        out.strides()[j] / dtype_to_size(out.dtype()); // stride is in bytes
    ir.push_back(stride);
    ctx.arg_to_strides.at(arg).push_back(stride);
  }
  auto [local_idxs, only_loads] =
      render_local_idxs(global_idx, ctx.arg_to_shape.at(arg), i);
  ir.insert(ir.end(), local_idxs.begin(), local_idxs.end());
  ctx.arg_to_idxs_to_load[arg] = only_loads;

  // first, render the accumulator (imm value of 0)
  auto acc = std::make_shared<ImmExpr>();
  acc->value = (is<Sum>(out.ad_node()->primitive()) ||
                is<Mean>(out.ad_node()->primitive()))
                   ? 0
                   : std::numeric_limits<float>::lowest();
  acc->dtype = out.dtype();
  acc->force_render = true;
  acc->name = "acc0";
  ir.push_back(acc);

  auto reduce_loops = std::vector<std::shared_ptr<ForStartExpr>>();
  for (int i = 0; i < reduce->axes().size(); i++) {
    auto axis = reduce->axes()[i] < 0 ? inputs[0].ndim() + reduce->axes()[i]
                                      : reduce->axes()[i];
    auto reduce_loop = std::make_shared<ForStartExpr>();
    auto start = std::make_shared<ImmExpr>();
    start->force_render = true;
    start->value = 0;
    reduce_loop->start = start;
    auto end = std::make_shared<ImmExpr>();
    end->value = inputs[0].shape()[axis];
    reduce_loop->end = end;
    auto step = std::make_shared<ImmExpr>();
    step->value = 1;
    reduce_loop->step = step;
    ir.push_back(start);
    ir.push_back(end);
    ir.push_back(step);
    ir.push_back(reduce_loop);
    reduce_loops.push_back(reduce_loop);
  }
  i = 0;

  // here, render the missing input idxs based on the reduce idx for each input
  for (auto &input : inputs) {
    if (is<Fill>(input.ad_node()->primitive())) {
      i++;
      continue;
    }
    int redidx = 0;
    // Render the inner idxs of the loop
    for (int x = 0; x < input.ndim(); x++) {
      if (!is_reduced(x)) {
        continue;
      }
      auto arg = ctx.tid_to_arg.at(input.id);
      auto [local_idxs_reduce, only_loads_reduce] = render_local_idxs(
          reduce_loops.at(redidx)->start, ctx.arg_to_shape.at(arg), i,
          [=](int i) { return i == x; });

      ir.insert(ir.end(), local_idxs_reduce.begin(), local_idxs_reduce.end());
      for (int j = 0; j < only_loads_reduce.size(); j++) {
        if (only_loads_reduce[j] != nullptr) {
          ctx.arg_to_idxs_to_load.at(arg)[j] = only_loads_reduce[j];
        }
      }
      redidx++;
    }
    i++;
  }

  // render inner
  std::vector<Tensor> x;
  graph_to_ir_inner(out.ad_node()->children()[0], x, ir, ctx, inputs);

  // acc += inner_ir[-1]

  auto acc_binop = std::make_shared<AccumExpr>();
  auto prim = out.ad_node()->primitive();
  acc_binop->op =
      (is<Sum>(prim) || is<Mean>(prim)) ? AccumOpKind::Add : AccumOpKind::Max;
  acc_binop->lhs = acc;
  acc_binop->rhs = ir.back();

  ir.push_back(acc_binop);

  // end loops
  for (int i = reduce_loops.size() - 1; i >= 0; i--) {
    auto reduce_loop = reduce_loops[i];
    auto for_end = std::make_shared<ForEndExpr>();
    for_end->for_start = reduce_loop;
    ir.push_back(for_end);
  }

  // if its mean, divide by total_reduced_elems
  if (is<Mean>(prim)) {
    auto div = std::make_shared<BinaryExpr>();
    div->op = BinaryOpKind::Div;
    div->lhs = acc;
    auto total_reduced_elems_imm = std::make_shared<ImmExpr>();
    total_reduced_elems_imm->value = total_reduced_elems;
    ir.push_back(total_reduced_elems_imm);
    div->rhs = total_reduced_elems_imm;
    ir.push_back(div);
    auto store_ir = render_store_idxs_for_expr(
        ctx.arg_to_idxs_to_load.at(arg), ctx.arg_to_strides.at(arg), arg, div);
    ir.insert(ir.end(), store_ir.begin(), store_ir.end());
  }
  // else, just store the acc
  else {
    auto store_ir = render_store_idxs_for_expr(
        ctx.arg_to_idxs_to_load.at(arg), ctx.arg_to_strides.at(arg), arg, acc);

    ir.insert(ir.end(), store_ir.begin(), store_ir.end());
  }
  // optim_ir_implace(ir);
  return {ir, ctx, used_inputs};
}

std::tuple<std::vector<std::shared_ptr<BaseExpr>>, IrBuilderContext,
           std::vector<bool>>
graph_to_ir(Tensor &out, std::vector<Tensor> marked_as_out,
            const std::vector<Tensor> &inputs) {
  // out = reduce(fn(inputs), axis=...)
  if (is_reduce_op(out.ad_node()->primitive())) {
    return graph_to_ir_reduce(out, inputs);
  }
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
  auto return_guard = render_return_guard(out.numel(), global_idx);
  ir.insert(ir.end(), return_guard.begin(), return_guard.end());

  // fill the ctx and ir with the input tensors
  int i = 0;
  std::vector<bool> used_inputs(inputs.size(), false);
  for (auto &input : inputs) {
    if (is<Fill>(input.ad_node()->primitive()) ||
        get_fill_value(input) != nullptr) {
      i++;
      // will be replaced by a constant
      continue;
    }
    used_inputs[i] = true;
    auto arg = std::make_shared<ArgExpr>();
    arg->dtype = input.dtype();
    ir.push_back(arg);
    ctx.tid_to_arg[input.id] = arg;
    // add to ctx.tid_to_shape and tid_to_strides
    ctx.arg_to_shape[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_strides[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_idxs_to_load[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_idxs_to_load[arg].resize(input.ndim());
    for (int j = 0; j < input.ndim(); j++) {
      auto shape = std::make_shared<ImmExpr>();
      shape->value = input.shape()[j];
      ir.push_back(shape);
      ctx.arg_to_shape.at(arg).push_back(shape);
      auto stride = std::make_shared<ImmExpr>();
      stride->value = input.strides()[j] /
                      dtype_to_size(input.dtype()); // stride is in bytes
      ir.push_back(stride);
      ctx.arg_to_strides.at(arg).push_back(stride);
    }
    auto [local_idxs, only_loads] =
        render_local_idxs(global_idx, ctx.arg_to_shape.at(arg), i);
    ir.insert(ir.end(), local_idxs.begin(), local_idxs.end());
    ctx.arg_to_idxs_to_load[arg] = only_loads;

    i++;
  }

  // same for the marked_as_out
  for (auto &input : marked_as_out) {
    if (is<Fill>(input.ad_node()->primitive())) {
      i++;
      // will be replaced by a constant
      continue;
    }
    auto arg = std::make_shared<ArgExpr>();
    arg->name = "out" + std::to_string(i);
    arg->dtype = input.dtype();
    ir.push_back(arg);
    ctx.tid_to_arg[input.id] = arg;
    // add to ctx.tid_to_shape and tid_to_strides
    ctx.arg_to_shape[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_strides[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_idxs_to_load[arg] = std::vector<std::shared_ptr<BaseExpr>>();
    ctx.arg_to_idxs_to_load[arg].resize(input.ndim());
    for (int j = 0; j < input.ndim(); j++) {
      auto shape = std::make_shared<ImmExpr>();
      shape->value = input.shape()[j];
      ir.push_back(shape);
      ctx.arg_to_shape.at(arg).push_back(shape);
      auto stride = std::make_shared<ImmExpr>();
      stride->value = input.strides()[j] /
                      dtype_to_size(input.dtype()); // stride is in bytes
      ir.push_back(stride);
      ctx.arg_to_strides.at(arg).push_back(stride);
    }
    auto [local_idxs, only_loads] =
        render_local_idxs(global_idx, ctx.arg_to_shape.at(arg), i);
    ir.insert(ir.end(), local_idxs.begin(), local_idxs.end());
    ctx.arg_to_idxs_to_load[arg] = only_loads;
    i++;
  }

  // now same but for the output tensor
  /*auto arg = std::make_shared<ArgExpr>();
  arg->name = "out" + std::to_string(i);
  arg->dtype = out.dtype();
  ir.push_back(arg);
  ctx.tid_to_arg[out.id] = arg;
  ctx.arg_to_shape[arg] = std::vector<std::shared_ptr<BaseExpr>>();
  ctx.arg_to_strides[arg] = std::vector<std::shared_ptr<BaseExpr>>();
  ctx.arg_to_idxs_to_load[arg] = std::vector<std::shared_ptr<BaseExpr>>();
  ctx.arg_to_idxs_to_load[arg].resize(out.ndim());
  for (int j = 0; j < out.ndim(); j++) {
    auto shape = std::make_shared<ImmExpr>();
    shape->value = out.shape()[j];
    ir.push_back(shape);
    ctx.arg_to_shape.at(arg).push_back(shape);
    auto stride = std::make_shared<ImmExpr>();
    stride->value =
        out.strides()[j] / dtype_to_size(out.dtype()); // stride is in bytes
    ir.push_back(stride);
    ctx.arg_to_strides.at(arg).push_back(stride);
  }
  auto [local_idxs, only_loads] =
      render_local_idxs(global_idx, ctx.arg_to_shape.at(arg), i);
  ir.insert(ir.end(), local_idxs.begin(), local_idxs.end());
  ctx.arg_to_idxs_to_load[arg] = only_loads; */

  graph_to_ir_inner(out, marked_as_out, ir, ctx, inputs);

  // now, at the end, add a store operation
  // we need to get the last ir item that is not a store
  /*std::shared_ptr<BaseExpr> last_ir_item;
  for (int i = ir.size() - 1; i >= 0; i--) {
    if (!is<StoreExpr>(ir[i])) {
      last_ir_item = ir[i];
      break;
    }
  }
  auto store_ir =
      render_store_idxs_for_expr(ctx.arg_to_idxs_to_load.at(arg),
                                 ctx.arg_to_strides.at(arg), arg, last_ir_item);


  ir.insert(ir.end(), store_ir.begin(), store_ir.end());*/

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
std::pair<std::string, std::string>
ir_to_cuda(std::vector<std::shared_ptr<BaseExpr>> &ir) {
  assign_names_to_ir(ir);
  // TODO -- handle this better
  bool is_reduce = false;
  for (auto &expr : ir) {
    if (is<AccumExpr>(expr)) {
      is_reduce = true;
      break;
    }
  }
  std::string kernel_name = "elementwise_kernel";
  if (is_reduce) {
    kernel_name = "reduce_kernel";
  }
  std::string res = render_fn_header("", ir) + "\n";
  res = "__global__ void " + kernel_name + "(" + res + ")";
  res += "{\t\n";
  std::map<std::shared_ptr<BaseExpr>, std::string> r;
  std::map<std::string, bool> rendered; // used for example not to render block
                                        // dim twice
  NameDatabase nmdb;
  int indent_level = 1;
  auto add_indent = [&indent_level]() {
    return std::string(indent_level * 2, ' ');
  };
  for (auto &expr : ir) {
    if (is<BinaryExpr>(expr)) {
      auto binop = as<BinaryExpr>(expr);
      if (!binop->force_render) {
        r[expr] = "(" +
                  binop_kind_to_str(binop->op, r[binop->lhs], r[binop->rhs]) +
                  ")";
      } else {
        r[expr] = binop->name;
        res += add_indent() + get_dtype_cpp_str(binop->dtype) + " " +
               binop->name + " = " +
               binop_kind_to_str(binop->op, r[binop->lhs], r[binop->rhs]) +
               ";\n";
      }
    } else if (is<UnaryExpr>(expr)) {
      auto unop = as<UnaryExpr>(expr);
      if (!unop->force_render) {
        r[expr] = unop_kind_to_str(unop->op) + "(" + r[unop->child] + ")";
      } else {
        r[expr] = unop->name;
        res += add_indent() + get_dtype_cpp_str(unop->dtype) + " " +
               unop->name + " = " + unop_kind_to_str(unop->op) + "(" +
               r[unop->child] + ");\n";
      }
    } else if (is<ImmExpr>(expr)) {
      auto imm = as<ImmExpr>(expr);
      auto val_to_str = [&imm]() {
        if (imm->dtype == DType::Int32) {
          return std::to_string((int)imm->value);
        } else if (imm->dtype == DType::Float32) {
          std::ostringstream oss;
          oss << std::fixed
              << std::setprecision(std::numeric_limits<float>::max_digits10)
              << imm->value;
          return oss.str();
        } else {
          PG_CHECK_RUNTIME(false, "Unsupported dtype");
        }
      };

      if (expr->force_render) {
        r[expr] = expr->name;
        res += add_indent() + get_dtype_cpp_str(imm->dtype) + " " + expr->name +
               " = " + val_to_str() + ";\n";
      } else {
        r[expr] = val_to_str();
      }

    } else if (is<ArgExpr>(expr)) {
      continue; // already rendered in the header
    } else if (is<BlockIdxExpr>(expr)) {
      auto block_idx = as<BlockIdxExpr>(expr);
      if (!rendered["bidx"]) {
        res += add_indent() + get_dtype_cpp_str(DType::Int32) + " " + "bidx" +
               " = " + "blockIdx.x" + ";\n";
        rendered["bidx"] = true;
      }
      r[expr] = "bidx";
    } else if (is<BlockDimExpr>(expr)) {
      auto block_dim = as<BlockDimExpr>(expr);
      if (!rendered["bdim"]) {
        res += add_indent() + get_dtype_cpp_str(DType::Int32) + " " + "bdim" +
               " = " + "blockDim.x" + ";\n";
        rendered["bdim"] = true;
      }
      r[expr] = "bdim";
    } else if (is<ThreadIdxExpr>(expr)) {
      auto thread_idx = as<ThreadIdxExpr>(expr);
      if (!rendered["tidx"]) {
        res += add_indent() + get_dtype_cpp_str(DType::Int32) + " " + "tidx" +
               " = " + "threadIdx.x" + ";\n";
        rendered["tidx"] = true;
      }
      r[expr] = "tidx";
    } else if (is<LoadExpr>(expr)) {
      auto load = as<LoadExpr>(expr);
      auto x = nmdb.get_with_prefix("load");
      r[expr] = x;
      res += add_indent() + get_dtype_cpp_str(load->dtype) + " " + x + " = " +
             load->child->name + "[" + r[load->idx] + "]" + ";\n";
    } else if (is<StoreExpr>(expr)) {
      auto store = as<StoreExpr>(expr);
      res += add_indent() + store->ptr->name + "[" + r[store->idx] +
             "] = " + r[store->value] + ";\n";
    } else if (is<IfStartExpr>(expr)) {
      auto if_start = as<IfStartExpr>(expr);
      res += add_indent() + "if (" + r[if_start->cond] + ") {\n";
      indent_level++;
    } else if (is<IfEndExpr>(expr)) {
      indent_level--;
      res += add_indent() + "}\n";
    } else if (is<ReturnExpr>(expr)) {
      res += add_indent() + "return;\n";
    } else if (is<TernaryExpr>(expr)) {
      auto ternop = as<TernaryExpr>(expr);
      r[expr] = "(" +
                ternop_kind_to_str(ternop->op, r[ternop->first],
                                   r[ternop->second], r[ternop->third]) +
                ")";
    } else if (is<ForStartExpr>(expr)) {
      auto for_start = as<ForStartExpr>(expr);
      // res += add_indent() + "#pragma unroll\n"; // prevent register
      // overloading -- in the future maybe 'search' for optimal unroll factor
      // ??
      res += add_indent() + "for (" + r[for_start->start] + "; " +
             r[for_start->start] + " < " + r[for_start->end] + "; " +
             r[for_start->start] + "+=" + r[for_start->step] + ") {\n";
      indent_level++;
    } else if (is<ForEndExpr>(expr)) {
      indent_level--;
      res += add_indent() + "}\n";
    } else if (is<AccumExpr>(expr)) {
      PG_CHECK_RUNTIME(as<AccumExpr>(expr)->lhs->force_render,
                       "Accum lhs must be forced to render");
      auto acc = as<AccumExpr>(expr);
      if (acc->op == AccumOpKind::Add) {
        res += add_indent() + r[acc->lhs] + " += " + r[acc->rhs] + ";\n";
      } else if (acc->op == AccumOpKind::Max) {
        res += add_indent() + r[acc->lhs] + " = max(" + r[acc->lhs] + ", " +
               r[acc->rhs] + ");\n";
      } else {
        PG_CHECK_RUNTIME(false, "Unsupported accum op");
      }
    } else {
      PG_CHECK_RUNTIME(false, "Unsupported expression: " + expr->expr_str());
    }
  }
  res += "}\n";
  return {res, kernel_name};
}
} // namespace ir

void Compiled::dispatch_cuda(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs) {
  // first, we need to gather ir
  using namespace ir;
  if (this->cached_fn == nullptr) {

    // firsts
    auto [ker, ker_name] = ir_to_cuda(this->ir);
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
    CUresult R = cuModuleGetFunction(&cuFunction, cuModule, ker_name.c_str());
    PG_CHECK_RUNTIME(R == CUDA_SUCCESS, "Failed to get function: got " +
                                            std::to_string(R) + " for kernel " +
                                            ker_name);

    PG_CHECK_RUNTIME(cuFunction != nullptr, "Failed to get function");
    // Store the function pointer in a void*
    void *function_ptr = reinterpret_cast<void *>(cuFunction);
    PG_CHECK_RUNTIME(function_ptr != nullptr, "Failed to get function pointer");
    // Clean up
    nvrtcDestroyProgram(&prog);
    delete[] ptx;
    this->cached_fn = function_ptr;
    this->_kername = ker_name;
  }
  // allocate each output
  for (auto &output : outputs) {
    output.view_ptr()->allocate();
  }
  // Prepare kernel arguments
  // Prepare grid and block dimensions
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  dim3 threads_per_block(prop.maxThreadsPerBlock / 2, 1, 1);
  size_t num_elements = outputs[0].numel();
  dim3 blocks_per_grid(
      (num_elements + threads_per_block.x - 1) / threads_per_block.x, 1, 1);
  std::vector<void *> kernel_args;
  for (const auto &input : inputs) {
    void *in_data = input.get_base_ptr();
    kernel_args.push_back(in_data);
  }
  for (auto &output : outputs) {

    void *out_data = output.get_base_ptr();
    kernel_args.push_back(out_data);
  }

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

  // check error again
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    PG_CHECK_RUNTIME(false,
                     "cuda error: " + std::string(cudaGetErrorString(error)));
  }
}

} // namespace pg