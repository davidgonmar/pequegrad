#include "ir.hpp"
#include <chrono>
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

// Function that takes a lambda comparing, and returns true if a backward graph
// visit makes it true
static bool
visit_with_condition(Tensor &t, std::function<bool(Tensor &)> f,
                     std::function<bool(Tensor &)> should_end_branch) {
  if (f(t)) {
    return true;
  }
  if (should_end_branch(t)) {
    return false;
  }
  for (auto &child : t.ad_node()->children()) {
    if (visit_with_condition(child, f, should_end_branch)) {
      return true;
    }
  }
  return false;
}

static Tensor &visit_until(Tensor &t, std::function<bool(Tensor &)> f,
                           std::function<bool(Tensor &)> should_end_branch) {
  if (f(t)) {
    return t;
  }
  if (should_end_branch(t)) {
    throw std::runtime_error("visit_until: not found");
  }
  for (auto &child : t.ad_node()->children()) {
    try {
      return visit_until(child, f, should_end_branch);
    } catch (const std::runtime_error &) {
      // Continue to the next child. Only throw if its root
    }
  }
  throw std::runtime_error("visit_until: not found");
}

static void visit_all_branches_until(Tensor &t,
                                     std::function<bool(Tensor &)> f) {
  if (f(t)) {
    return;
  }
  for (auto &child : t.ad_node()->children()) {
    visit_all_branches_until(child, f);
  }
}

static std::vector<Tensor> visit_all_until(Tensor &t,
                                           std::function<bool(Tensor &)> f) {
  std::vector<Tensor> res;
  if (f(t)) {
    res.push_back(t);
    return res;
  }
  for (auto &child : t.ad_node()->children()) {
    auto res_ = visit_all_until(child, f);
    res.insert(res.end(), res_.begin(), res_.end());
  }
  return res;
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
                  IrBuilderContext &ctx, const std::vector<Tensor> &orig_inputs,
                  std::shared_ptr<BaseExpr> reduce_result = nullptr,
                  std::shared_ptr<Tensor> reduced_tensor = nullptr) {
  auto prim = out.ad_node()->primitive();
  if (reduce_result != nullptr) {
    PG_CHECK_RUNTIME(reduced_tensor != nullptr,
                     "reduced_tensor is null. Out: " + out.str());
  }
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
    auto ir_load = render_load_idxs_for_expr(idxs_to_load, strides, arg, ctx);
    // irload is a vector
    ir.insert(ir.end(), ir_load.begin(), ir_load.end());
    return ir_load.back();
  }

  if (reduced_tensor != nullptr && reduced_tensor->id == out.id) {
    PG_CHECK_RUNTIME(reduce_result != nullptr,
                     "reduce_result is null. Out: " + out.str());
    return reduce_result;
  }

  // first recursively the input tensors
  std::vector<std::shared_ptr<BaseExpr>> inputs;

  for (auto &input : out.ad_node()->children()) {
    auto ir_ = graph_to_ir_inner(input, marked_as_out, ir, ctx, orig_inputs,
                                 reduce_result, reduced_tensor);
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
          ctx.arg_to_idxs_to_load.at(arg), strides, arg, binop, ctx);
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
          ctx.arg_to_idxs_to_load.at(arg), strides, arg, unop, ctx);
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
          ctx.arg_to_idxs_to_load.at(arg), strides, arg, ternop, ctx);
      ir.insert(ir.end(), store_ir.begin(), store_ir.end());
    }
    return ternop;
  }

  std::string inps_str = "";
  for (auto &inp : orig_inputs) {
    inps_str += inp.str() + " ";
  }
  throw std::runtime_error(
      "Bad schedule. Not an input and not a supported op: out: " + out.str() +
      " reduced_tensor is null: " +
      (reduced_tensor == nullptr ? "null" : reduced_tensor->str()) +
      " inputs: " + inps_str);
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
    ir_item_t gidx, ir_t shapes, int input_idx, bool is_contiguous = false,
    l choose_which_idxs_to_load = default_choose_which_idxs_to_load) {
  ir_t res;
  ir_t only_loads = ir_t();
  // if is contigous, only return global idx
  if (is_contiguous) {
    only_loads.push_back(gidx);
    return {res, only_loads};
  }
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
  std::vector<Tensor> inputs_visited;
  Tensor &reduced = visit_until(
      out,
      [&inputs](Tensor &t) {
        return is_reduce_op(t.ad_node()->primitive()) &&
               std::find_if(inputs.begin(), inputs.end(),
                            [&t](const Tensor &input) {
                              return t.id == input.id;
                            }) == inputs.end();
      },
      [&inputs](Tensor &t) {
        return std::find_if(inputs.begin(), inputs.end(),
                            [&t](const Tensor &input) {
                              return t.id == input.id;
                            }) != inputs.end();
      });

  std::vector<Tensor> reduced_producer_leafs =
      visit_all_until(reduced, [&inputs](Tensor &t) {
        // return if it is in inputs
        return std::any_of(inputs.begin(), inputs.end(),
                           [&t](const Tensor &input) {
                             return t.id == input.id && !get_fill_value(input);
                           });
      });

  // make reduced_producer_leafs unique
  std::unordered_map<int, bool> reduced_producer_leafs_map;
  std::vector<Tensor> unique_reduced_producer_leafs;
  for (auto &leaf : reduced_producer_leafs) {
    if (reduced_producer_leafs_map.find(leaf.id) ==
        reduced_producer_leafs_map.end()) {
      reduced_producer_leafs_map[leaf.id] = true;
      unique_reduced_producer_leafs.push_back(leaf);
    }
  }
  reduced_producer_leafs = unique_reduced_producer_leafs;

  // Start of Selection
  auto reduced_depends_on_input = [&reduced_producer_leafs](const Tensor &t) {
    return std::any_of(reduced_producer_leafs.begin(),
                       reduced_producer_leafs.end(),
                       [&t](const Tensor &leaf) { return t.id == leaf.id; });
  };

  std::shared_ptr<Reduce> reduce = as<Reduce>(reduced.ad_node()->primitive());
  axes_t axes = reduce->axes();
  for (int xx = 0; xx < axes.size(); xx++) {
    axes[xx] = axes[xx] < 0 ? reduced.ad_node()->children()[0].ndim() + axes[xx]
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
  std::vector<int> already_used; // used to dedupe inputs
  std::vector<bool> used_inputs(inputs.size(), false);
  for (auto &input : inputs) {
    if (is<Fill>(input.ad_node()->primitive()) ||
        get_fill_value(input) != nullptr) {
      i++;
      // will be replaced by a constant
      continue;
    }
    if (std::find(already_used.begin(), already_used.end(), input.id) !=
        already_used.end()) {
      i++;
      continue;
    }
    already_used.push_back(input.id);
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
    auto [local_idxs, only_loads] = render_local_idxs(
        global_idx, ctx.arg_to_shape.at(arg), i, false, [=](int i) {
          return !is_reduced(i) || !reduced_depends_on_input(input);
        });
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
  acc->value = (is<Sum>(reduce) || is<Mean>(reduce))
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
    if (!used_inputs[i]) {
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
          reduce_loops.at(redidx)->start, ctx.arg_to_shape.at(arg), i, false,
          [=](int i) { return i == x && reduced_depends_on_input(input); });

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
  graph_to_ir_inner(reduced.ad_node()->children()[0], x, ir, ctx, inputs);
  // acc += inner_ir[-1]

  auto acc_binop = std::make_shared<AccumExpr>();
  auto prim = reduce;
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

  auto reduce_result = std::shared_ptr<BaseExpr>();
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
    reduce_result = div;
    // auto store_ir = render_store_idxs_for_expr(
    // ctx.arg_to_idxs_to_load.at(arg), ctx.arg_to_strides.at(arg), arg, div);
    // ir.insert(ir.end(), store_ir.begin(), store_ir.end());
  }
  // else, just store the acc
  else {
    reduce_result = acc;
    // auto store_ir = render_store_idxs_for_expr(
    // ctx.arg_to_idxs_to_load.at(arg), ctx.arg_to_strides.at(arg), arg, acc);

    // ir.insert(ir.end(), store_ir.begin(), store_ir.end());
  }
  if (out.id == reduced.id) {
    auto _arg = ctx.tid_to_arg.at(out.id);
    auto store_ir = render_store_idxs_for_expr(ctx.arg_to_idxs_to_load.at(_arg),
                                               ctx.arg_to_strides.at(_arg),
                                               _arg, reduce_result, ctx);
    ir.insert(ir.end(), store_ir.begin(), store_ir.end());
    return {ir, ctx, used_inputs};
  }

  std::shared_ptr<Tensor> reduced_tensor = std::make_shared<Tensor>(reduced);
  graph_to_ir_inner(out, x, ir, ctx, inputs, reduce_result, reduced_tensor);

  // The result is in the last element of the ir
  // render a store for the result
  auto _arg = ctx.tid_to_arg.at(out.id);
  auto store_ir = render_store_idxs_for_expr(ctx.arg_to_idxs_to_load.at(_arg),
                                             ctx.arg_to_strides.at(_arg), _arg,
                                             ir.back(), ctx);
  ir.insert(ir.end(), store_ir.begin(), store_ir.end());

  // optim_ir_implace(ir);
  return {ir, ctx, used_inputs};
}

std::tuple<std::vector<std::shared_ptr<BaseExpr>>, IrBuilderContext,
           std::vector<bool>>
graph_to_ir(Tensor &out, std::vector<Tensor> marked_as_out,
            const std::vector<Tensor> &inputs) {
  // out = fn(reduce(fn(inputs), axis=...))
  if (visit_with_condition(
          out,
          [&inputs](const Tensor &t) {
            // if there is a reduce op that is not an input
            return is_reduce_op(t.ad_node()->primitive()) &&
                   std::find_if(inputs.begin(), inputs.end(),
                                [&t](const Tensor &input) {
                                  return t.id == input.id;
                                }) == inputs.end();
          },
          [&inputs](const Tensor &t) {
            // if there is an input
            return std::find_if(inputs.begin(), inputs.end(),
                                [&t](const Tensor &input) {
                                  return t.id == input.id;
                                }) != inputs.end();
          })) {
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
    auto [local_idxs, only_loads] = render_local_idxs(
        global_idx, ctx.arg_to_shape.at(arg), i, input.is_contiguous());
    ir.insert(ir.end(), local_idxs.begin(), local_idxs.end());
    ctx.arg_to_idxs_to_load[arg] = only_loads;
    ctx.arg_to_is_contiguous[arg] = input.is_contiguous();
    i++;
  }

  // same for the marked_as_out
  for (auto &input : marked_as_out) {
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
    auto [local_idxs, only_loads] = render_local_idxs(
        global_idx, ctx.arg_to_shape.at(arg), i, input.is_contiguous());
    ir.insert(ir.end(), local_idxs.begin(), local_idxs.end());
    ctx.arg_to_idxs_to_load[arg] = only_loads;
    ctx.arg_to_is_contiguous[arg] = input.is_contiguous();
    i++;
  }
  graph_to_ir_inner(out, marked_as_out, ir, ctx, inputs);
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
    res += get_dtype_cpp_str(arg->dtype) + " * __restrict__ " + arg->name;
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
  // add random id
  kernel_name += "_" + std::to_string(rand());
  std::string res = render_fn_header("", ir) + "\n";
  res = "__global__ void __launch_bounds__(1024, 1) " + kernel_name + "(" +
        res + ")";
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
      res += add_indent() + "#pragma unroll\n"; // prevent register
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
  // allocate each output
  for (auto &output : outputs) {
    output.view_ptr()->allocate();
  }
  // first, we need to gather ir
  using namespace ir;
  if (this->jit_kernel == nullptr) {

    // firsts
    auto [ker, ker_name] = ir_to_cuda(this->ir);
    this->jit_kernel = std::make_shared<CudaKernel>(ker_name, ker);
    if (getenv("PG_KERNEL_DB") != nullptr) {
      // print the kernel to stdout
      std::cout << ker << std::endl;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    auto threads_per_block = prop.maxThreadsPerBlock;
    auto blocks_per_grid =
        (outputs[0].numel() + threads_per_block - 1) / threads_per_block;
    // now do autotuning the threads per block
    double best_time = std::numeric_limits<double>::max();
    int best_threads_per_block = 0;

    for (int i = 2; i <= prop.maxThreadsPerBlock; i *= 2) {
      int curr_tpb = i;
      double total_time = 0.0;
      int num_repeats =
          10; // Number of times to repeat the kernel launch for better timing

      for (int j = 0; j < num_repeats; ++j) {
        // chrono
        auto start = std::chrono::high_resolution_clock::now();

        // launch
        dim3 threads_per_block(curr_tpb);
        size_t num_elements = outputs[0].numel();
        dim3 blocks_per_grid((num_elements + curr_tpb - 1) / curr_tpb);
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
        auto casted_to_cuda_kernel =
            std::dynamic_pointer_cast<CudaKernel>(this->jit_kernel);
        if (casted_to_cuda_kernel == nullptr) {
          throw std::runtime_error("Kernel is not a CudaKernel");
        }
        casted_to_cuda_kernel->set_blocks_per_grid(blocks_per_grid.x);
        casted_to_cuda_kernel->set_threads_per_block(threads_per_block.x);
        this->jit_kernel->launch(kernel_args_ptrs);
        // sync and end chrono
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        total_time += time;
      }

      double avg_time = total_time / num_repeats;
      if (avg_time < best_time) {
        best_time = avg_time;
        best_threads_per_block = curr_tpb;
      }
    }

    auto casted = std::dynamic_pointer_cast<CudaKernel>(this->jit_kernel);
    casted->set_blocks_per_grid(blocks_per_grid);
    casted->set_threads_per_block(threads_per_block);
  }

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
  this->jit_kernel->launch(kernel_args_ptrs);
}

} // namespace pg