#include "ir.hpp"
#include "new_ir.hpp"
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
std::string get_dtype_cpp_str(DType dtype) {
  switch (dtype) {
  case DType::Float32:
    return "float";
  case DType::Int32:
    return "int";
  case DType::Float16:
    return "half";
  case DType::Float64:
    return "double";
  default:
    throw std::runtime_error("Unsupported dtype in get_dtype_cpp_str");
  }
}

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

static newir::BinaryOpKind op_to_binop_kind_new(ADPrimitive &prim) {
  if (is<Add>(prim)) {
    return newir::BinaryOpKind::Add;
  } else if (is<Sub>(prim)) {
    return newir::BinaryOpKind::Sub;
  } else if (is<Mul>(prim)) {
    return newir::BinaryOpKind::Mul;
  } else if (is<Div>(prim)) {
    return newir::BinaryOpKind::Div;
  } else if (is<Gt>(prim)) {
    return newir::BinaryOpKind::Gt;
  } else if (is<Lt>(prim)) {
    return newir::BinaryOpKind::Lt;
  } else if (is<Max>(prim)) {
    return newir::BinaryOpKind::Max;
  } else if (is<Eq>(prim)) {
    return newir::BinaryOpKind::Eq;
  } else if (is<Pow>(prim)) {
    return newir::BinaryOpKind::Pow;
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
static newir::UnaryOpKind op_to_unaryop_kind_new(ADPrimitive &prim) {
  if (is<Log>(prim)) {
    return newir::UnaryOpKind::Log;
  } else if (is<Exp>(prim)) {
    return newir::UnaryOpKind::Exp;
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
static newir::TernaryOpKind op_to_ternaryop_kind_new(ADPrimitive &prim) {
  if (is<Where>(prim)) {
    return newir::TernaryOpKind::Where;
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

namespace n = newir;
class PrinterContext {
public:
  int indent = 0;
  std::string str = "";
  void add_indent() { indent++; }
  void remove_indent() { indent--; }
  void operator<<(std::string s) {
    if (str.size() > 0 && str.back() == '\n') {
      for (int i = 0; i < indent; i++) {
        str += " ";
      }
    }
    str += s;
  }
  std::map<std::shared_ptr<n::BaseExpr>, std::string> expr_to_str;
  int varidx = 0;
  void savevar(std::shared_ptr<n::BaseExpr> expr) {
    expr_to_str[expr] = "t" + std::to_string(varidx++);
  }
  void savevarexpr(std::shared_ptr<n::BaseExpr> expr, std::string s) {
    expr_to_str[expr] = s;
  }
  std::map<std::shared_ptr<n::LoopExpr>, int> loop_depth;
};

void print_new_ir(std::shared_ptr<n::BaseExpr> expr, PrinterContext &ctx);

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

void print_function(std::shared_ptr<n::FunctionExpr> func) {
  PrinterContext ctx;
  // fill context with loop depths
  std::shared_ptr<n::BaseExpr> block = func;
  while (block != nullptr) {
    // if it is not a func or a loop, break
    if (block->expr_str() != "FunctionExpr" &&
        block->expr_str() != "LoopExpr") {
      break;
    }
    // else, get its body
    std::vector<std::shared_ptr<n::BaseExpr>> body;
    if (block->expr_str() == "FunctionExpr") {
      body = std::static_pointer_cast<n::FunctionExpr>(block)->body;
    } else {
      body = std::static_pointer_cast<n::LoopExpr>(block)->body;
    }
    // for each expr in the body, if it is a loop, add it to the loop_depth
    bool found = false;
    for (auto &expr : body) {
      if (expr->expr_str() == "LoopExpr") {
        ctx.loop_depth[std::static_pointer_cast<n::LoopExpr>(expr)] =
            ctx.loop_depth.size();
        // ASSUME only one loop per block
        block = expr;
        found = true;
        break;
      }
    }
    if (!found) {
      break;
    }
  }

  // preamble with args
  ctx << "func " + func->name + "(";
  int argidx = 0;
  for (auto &arg : func->args) {
    ctx << "arg" + std::to_string(argidx) + "<" +
               get_dtype_cpp_str(arg->dtype) + ">";
    if (argidx != func->args.size() - 1) {
      ctx << ", ";
    }
    argidx++;
  }
  ctx << ") {\n";
  ctx.add_indent();
  // body
  for (auto &expr : func->body) {
    print_new_ir(expr, ctx);
  }
  ctx.remove_indent();
  ctx << "}\n";
  std::cout << ctx.str;
}

class IrNewBuilderContext {
public:
  std::map<int, std::shared_ptr<n::ArgExpr>> tid_to_arg;
  std::map<int, std::shared_ptr<n::BaseExpr>> dim_to_idx;
  std::map<int, std::shared_ptr<n::LoopExpr>> dim_to_loop;
  std::map<std::shared_ptr<n::ArgExpr>, std::vector<int>> arg_to_strides;
};

std::shared_ptr<n::BaseExpr> inner(Tensor &curr,
                                   const std::vector<Tensor> &inputs,
                                   IrNewBuilderContext ctx) {
  // if it is in inputs, return a load bound to all the idx of all the loops
  bool is_input =
      std::find_if(inputs.begin(), inputs.end(), [&curr](const Tensor &t) {
        return t.id == curr.id;
      }) != inputs.end();

  if (is_input) {
    auto arg = ctx.tid_to_arg.at(curr.id);
    // at the moment, just sum all the loops
    std::shared_ptr<n::BaseExpr> sum =
        ctx.dim_to_idx.at(0) * n::immint(ctx.arg_to_strides.at(arg)[0]);
    for (int i = 1; i < curr.ndim(); i++) {
      auto idx = ctx.dim_to_idx.at(i);
      sum = sum + (idx * n::immint(ctx.arg_to_strides.at(arg)[i]));
    }
    auto load = std::make_shared<n::LoadExpr>(arg, sum);
    ctx.dim_to_loop.at(curr.ndim() - 1)->body.push_back(load);
    return load;
  }
  // if binary ops, recurse on the children
  auto prim = curr.ad_node()->primitive();
  if (is_binary_op(prim)) {
    auto binop = std::make_shared<n::BinaryExpr>();
    binop->op = op_to_binop_kind_new(*prim);
    binop->lhs = inner(curr.ad_node()->children()[0], inputs, ctx);
    binop->rhs = inner(curr.ad_node()->children()[1], inputs, ctx);
    ctx.dim_to_loop.at(curr.ndim() - 1)->body.push_back(binop);
    return binop;
  }
  if (is_unary_op(prim)) {
    auto unop = std::make_shared<n::UnaryExpr>();
    unop->op = op_to_unaryop_kind_new(*prim);
    unop->src = inner(curr.ad_node()->children()[0], inputs, ctx);
    ctx.dim_to_loop.at(curr.ndim() - 1)->body.push_back(unop);
    return unop;
  }
  if (is_ternary_op(prim)) {
    auto ternop = std::make_shared<n::TernaryExpr>();
    ternop->op = op_to_ternaryop_kind_new(*prim);
    ternop->first = inner(curr.ad_node()->children()[0], inputs, ctx);
    ternop->second = inner(curr.ad_node()->children()[1], inputs, ctx);
    ternop->third = inner(curr.ad_node()->children()[2], inputs, ctx);
    ctx.dim_to_loop.at(curr.ndim() - 1)->body.push_back(ternop);
    return ternop;
  }
  throw std::runtime_error("Unsupported op");
}
void graph_to_ir_new(Tensor &out, std::vector<Tensor> marked_as_out,
                     const std::vector<Tensor> &inputs) {
  namespace n = newir;
  // the result will be a linear IR
  std::vector<std::shared_ptr<n::BaseExpr>> ir;
  IrNewBuilderContext ctx;

  std::vector<std::shared_ptr<n::ArgExpr>> args;
  int i = 0;
  for (auto &input : inputs) {
    args.push_back(std::make_shared<n::ArgExpr>(input.dtype(), i++));
    ctx.tid_to_arg[input.id] = args.back();
    auto str = input.strides();
    auto strcopy = std::vector<int>(str.begin(), str.end());
    for (int i = 0; i < strcopy.size(); i++) {
      strcopy[i] = strcopy[i] / dtype_to_size(input.dtype());
    }
    ctx.arg_to_strides[args.back()] = strcopy;
  }
  std::shared_ptr<n::FunctionExpr> func =
      std::make_shared<n::FunctionExpr>("kernel", args);
  // for each element of the shape, create a parallel for loop
  std::map<int, std::shared_ptr<n::LoopExpr>> dim_to_loop;
  for (int i = 0; i < out.ndim(); i++) {
    auto start = std::make_shared<n::ImmExpr>(DType::Int32, 0);
    auto end = std::make_shared<n::ImmExpr>(DType::Int32, out.shape()[i]);
    auto step = std::make_shared<n::ImmExpr>(DType::Int32, 1);
    auto loop = std::make_shared<n::LoopExpr>(n::LoopOpKind::Parallel, start,
                                              end, step);
    auto bidx = std::make_shared<n::BoundIdxExpr>(loop);
    dim_to_loop[i] = loop;
    // if it is the first loop, add it to the function body
    if (i == 0) {
      func->body.push_back(loop);
    }
    // else, add it to the body of the previous loop
    else {
      dim_to_loop[i - 1]->body.push_back(loop);
    }
    ctx.dim_to_idx[i] = bidx;
    ctx.dim_to_loop[i] = loop;
  }
  // add to output
  ctx.tid_to_arg[out.id] = std::make_shared<n::ArgExpr>(out.dtype(), i);
  auto item = inner(out, inputs, ctx);

  // add store to the output, in the last loop
  // render the store as a sum of all the idxs
  auto sum = ctx.dim_to_idx.at(0);
  for (int i = 1; i < out.ndim(); i++) {
    auto binop = std::make_shared<n::BinaryExpr>();
    binop->op = n::BinaryOpKind::Add;
    binop->lhs = sum;
    binop->rhs = ctx.dim_to_idx.at(i);
    sum = binop;
  }
  auto store =
      std::make_shared<n::StoreExpr>(ctx.tid_to_arg.at(out.id), sum, item);
  dim_to_loop[out.ndim() - 1]->body.push_back(store);
  // print the function
  print_function(func);
}

std::string operator<<(std::string &s, const char *c) {
  s += c;
  return s;
}
namespace n = newir;

void print_loop(std::shared_ptr<n::LoopExpr> loop, PrinterContext &ctx);
void print_load(std::shared_ptr<n::LoadExpr> load, PrinterContext &ctx);
void print_new_ir(std::shared_ptr<n::BaseExpr> expr, PrinterContext &ctx) {
  if (ctx.expr_to_str.find(expr) != ctx.expr_to_str.end()) {
    // ctx << ctx.expr_to_str.at(expr);
    return;
  }
  std::stringstream ss;
  std::string exprstr = expr->expr_str();
  if (exprstr == "LoopExpr") {
    print_loop(std::static_pointer_cast<n::LoopExpr>(expr), ctx);
    return;
  } else if (exprstr == "ImmExpr") {
    auto ex = std::static_pointer_cast<n::ImmExpr>(expr);
    double val = ex->value;
    if (ex->dtype == DType::Float32) {
      ss << std::to_string((float)val);
    } else if (ex->dtype == DType::Int32) {
      ss << std::to_string((int)val);
    } else if (ex->dtype == DType::Float16) {
      ss << std::to_string((float)val);
    }
    ctx.savevarexpr(expr, ss.str());
    return;
  } else if (exprstr == "LoadExpr") {
    auto load = std::static_pointer_cast<n::LoadExpr>(expr);
    ss << "load(";
    // print_new_ir(load->src, ctx);
    PG_CHECK_RUNTIME(load->src->expr_str() == "ArgExpr",
                     "load src is not an arg");
    ss << "arg" + std::to_string(
                      std::static_pointer_cast<n::ArgExpr>(load->src)->idx);
    ss << ", ";
    print_new_ir(load->idx, ctx);
    ss << ctx.expr_to_str.at(load->idx);
    ss << ")";
  } else if (exprstr == "BoundIdxExpr") {
    ss << "idx"
       << std::to_string(ctx.loop_depth.at(
              std::static_pointer_cast<n::BoundIdxExpr>(expr)->loop));
    ctx.savevarexpr(expr, ss.str());
    return;
  } else if (exprstr == "ReturnExpr") {
    ss << "return ";
    print_new_ir(std::static_pointer_cast<n::ReturnExpr>(expr)->value, ctx);
    ss << "\n";
  } else if (exprstr == "FunctionExpr") {
    print_function(std::static_pointer_cast<n::FunctionExpr>(expr));
  } else if (exprstr.find("BinaryExpr") == 0) {
    auto binop = std::static_pointer_cast<n::BinaryExpr>(expr);
    ss << "(";
    print_new_ir(binop->lhs, ctx);
    ss << ctx.expr_to_str.at(binop->lhs);
    ss << " ";
    switch (binop->op) {
    case n::BinaryOpKind::Add:
      ss << "+";
      break;
    case n::BinaryOpKind::Sub:
      ss << "-";
      break;
    case n::BinaryOpKind::Mul:
      ss << "*";
      break;
    case n::BinaryOpKind::Div:
      ss << "/";
      break;
    case n::BinaryOpKind::Gt:
      ss << ">";
      break;
    case n::BinaryOpKind::Lt:
      ss << "<";
      break;

    case n::BinaryOpKind::Max:
      ss << "max";
      break;
    case n::BinaryOpKind::Eq:
      ss << "==";
      break;
    case n::BinaryOpKind::Pow:
      ss << "pow";
      break;
    default:
      throw std::runtime_error("Unsupported binary op");
    }
    ss << " ";
    print_new_ir(binop->rhs, ctx);
    ss << ctx.expr_to_str.at(binop->rhs);
    ss << ")";
  } else if (exprstr == "UnaryExpr") {
    auto unop = std::static_pointer_cast<n::UnaryExpr>(expr);
    switch (unop->op) {
    case n::UnaryOpKind::Log:
      ss << "log(";
      break;
    case n::UnaryOpKind::Exp:
      ss << "exp(";
      break;
    default:
      throw std::runtime_error("Unsupported unary op");
    }
    print_new_ir(unop->src, ctx);
    ss << ")";
  } else if (exprstr == "TernaryExpr") {
    auto ternop = std::static_pointer_cast<n::TernaryExpr>(expr);
    ss << "where(";
    print_new_ir(ternop->first, ctx);
    ss << ", ";
    print_new_ir(ternop->second, ctx);
    ss << ", ";
    print_new_ir(ternop->third, ctx);
    ss << ")";
  } else if (exprstr == "StoreExpr") {
    auto store = std::static_pointer_cast<n::StoreExpr>(expr);
    ss << "store(";
    print_new_ir(store->dst, ctx);
    ss << ctx.expr_to_str.at(store->dst);
    ss << ", ";
    print_new_ir(store->idx, ctx);
    ss << ctx.expr_to_str.at(store->idx);
    ss << ", ";
    print_new_ir(store->src, ctx);
    ss << ctx.expr_to_str.at(store->src);
    ss << ")";
    ctx << ss.str();
    return;
  } else if (exprstr == "ArgExpr") {
    ss << "arg" +
              std::to_string(std::static_pointer_cast<n::ArgExpr>(expr)->idx);
    ctx.savevarexpr(expr, ss.str());
    return;
  } else {
    throw std::runtime_error("Unsupported expr: " + exprstr);
  }

  ctx.savevar(expr);
  ctx << ctx.expr_to_str.at(expr) + " = " + ss.str() + "\n";
}
// Fixing the syntax of the operator<< overload for appending a char array to a
// string

void print_loop(std::shared_ptr<n::LoopExpr> loop, PrinterContext &ctx) {
  ctx << "for (";
  print_new_ir(loop->start, ctx);
  ctx << ctx.expr_to_str.at(loop->start) + "; ";
  print_new_ir(loop->end, ctx);
  ctx << ctx.expr_to_str.at(loop->end) + "; ";
  print_new_ir(loop->step, ctx);
  ctx << ctx.expr_to_str.at(loop->step);
  ctx << ")->" + ("idx" + std::to_string(ctx.loop_depth.at(loop))) + " {\n";
  ctx.add_indent();
  for (auto &expr : loop->body) {
    print_new_ir(expr, ctx);
  }
  ctx.remove_indent();
  ctx << "}\n";
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
  cmp->op = BinaryOpKind::Lte;
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

#define NEWIR 0

std::tuple<std::vector<std::shared_ptr<BaseExpr>>, IrBuilderContext,
           std::vector<bool>>
graph_to_ir(Tensor &out, std::vector<Tensor> marked_as_out,
            const std::vector<Tensor> &inputs) {
#if NEWIR
  graph_to_ir_new(out, marked_as_out, inputs);
  throw std::runtime_error("Not implemented");
#endif
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
                                     std::string rhs, DType dtype) {
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
  case BinaryOpKind::Lte:
    return lhs + " <= " + rhs;
  case BinaryOpKind::Max:
    return "max(" + lhs + ", " + rhs + ")";
  case BinaryOpKind::Mod:
    return lhs + " % " + rhs;
  case BinaryOpKind::Eq:
    return lhs + " == " + rhs;
  case BinaryOpKind::Pow: // todo - differentiate between int and float pow
    return "pow(" + lhs + ", " + rhs + ")";
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
    res += (i != args.size() - 1 ? "const " : "") +
           get_dtype_cpp_str(arg->dtype) + " * __restrict__ " + arg->name;
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
  // include fp16
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
                  binop_kind_to_str(binop->op, r[binop->lhs], r[binop->rhs], binop->dtype) +
                  ")";
      } else {
        r[expr] = binop->name;
        res += add_indent() + get_dtype_cpp_str(binop->dtype) + " " +
               binop->name + " = " +
               binop_kind_to_str(binop->op, r[binop->lhs], r[binop->rhs], binop->dtype) +
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
          oss << std::fixed;
          if (std::isinf(imm->value)) {
            if (imm->value < 0) {
              oss << "-__int_as_float(0x7f800000)";
            } else {
              oss << "__int_as_float(0x7f800000)";
            }
          } else {
            oss << std::setprecision(std::numeric_limits<float>::max_digits10)
                << imm->value << "f";
          }
          return oss.str();
        } else if (imm->dtype == DType::Float16) {

          return "__float2half(" + std::to_string(imm->value) + ")";
        } else {
          throw std::runtime_error("Unsupported dtype in imm");
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
      /*res += add_indent() + get_dtype_cpp_str(load->dtype) + " " + x + " = " +
             load->child->name + "[" + r[load->idx] + "]" + ";\n"; */
      // instead, use __ldg
      res += add_indent() + get_dtype_cpp_str(load->dtype) + " " + x + " = " +
             "__ldg(" + load->child->name + " + " + r[load->idx] + ")" + ";\n";
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
    auto threads_per_block = prop.maxThreadsPerBlock / 2;
    auto blocks_per_grid =
        (outputs[0].numel() + threads_per_block - 1) / threads_per_block;
    // now do autotuning the threads per block
    double best_time = std::numeric_limits<double>::max();

    auto casted = std::dynamic_pointer_cast<CudaKernel>(this->jit_kernel);
    casted->set_threads_per_block(threads_per_block);
    casted->set_blocks_per_grid(blocks_per_grid);
  }

  std::vector<void *> kernel_args;
  for (auto &input : inputs) {
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