#include "expr.hpp"

namespace pg {
// returns a list with the inputs (leafs with LOAD operation)
// the map param is used to store the leafs -> tensor mapping
std::shared_ptr<AstExpr> get_ast_expr(
    Tensor &curr,
    std::map<std::shared_ptr<AstLoadExpr>, std::shared_ptr<Tensor>> &memo) {
  std::shared_ptr<ADPrimitive> prim = curr.ad_node().primitive();
  if (is<Log>(prim)) {
    auto expr = std::make_shared<AstUnaryExpr>();
    expr->op = AstUnaryOp::Log;
    expr->child = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  if (is<Exp>(prim)) {
    auto expr = std::make_shared<AstUnaryExpr>();
    expr->op = AstUnaryOp::Exp;
    expr->child = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  if (is<Add>(prim)) {
    auto expr = std::make_shared<AstBinaryExpr>();
    expr->op = AstBinaryOp::Add;
    expr->lhs = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->rhs = get_ast_expr(curr.ad_node().children()[1], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  if (is<Mul>(prim)) {
    auto expr = std::make_shared<AstBinaryExpr>();
    expr->op = AstBinaryOp::Mul;
    expr->lhs = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->rhs = get_ast_expr(curr.ad_node().children()[1], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  // print primitive
  // else, it's a load
  auto expr = std::make_shared<AstLoadExpr>();
  expr->name = "in" + std::to_string(curr.id);
  expr->dtype = curr.dtype();
  expr->shape = curr.shape();
  memo[expr] = std::make_shared<Tensor>(curr);
  return expr;
}

std::vector<std::shared_ptr<AstLoadExpr>>
get_leafs(std::shared_ptr<AstExpr> node) {
  using recurse_lambda_t = std::function<void(std::shared_ptr<AstExpr>)>;
  std::vector<std::shared_ptr<AstLoadExpr>> leafs;
  recurse_lambda_t recurse = [&](std::shared_ptr<AstExpr> node) {
    if (std::dynamic_pointer_cast<AstLoadExpr>(node)) {
      leafs.push_back(std::dynamic_pointer_cast<AstLoadExpr>(node));
    }
    if (std::dynamic_pointer_cast<AstUnaryExpr>(node)) {
      auto unary = std::dynamic_pointer_cast<AstUnaryExpr>(node);
      return recurse(unary->child);
    }
    if (std::dynamic_pointer_cast<AstBinaryExpr>(node)) {
      auto binary = std::dynamic_pointer_cast<AstBinaryExpr>(node);
      recurse(binary->lhs);
      recurse(binary->rhs);
    }
    if (std::dynamic_pointer_cast<AstStoreExpr>(node)) {
      auto store = std::dynamic_pointer_cast<AstStoreExpr>(node);
      return recurse(store->value);
    }
  };
  recurse(node);
  return leafs;
}

void fuse(Tensor &out) {
  std::map<std::shared_ptr<AstLoadExpr>, std::shared_ptr<Tensor>> memo;
  std::shared_ptr<AstExpr> ast = get_ast_expr(out, memo);
  //

  // Add a store operation after ast
  std::shared_ptr<AstStoreExpr> store = std::make_shared<AstStoreExpr>();
  store->name = "out";
  store->value = ast;
  store->dtype = out.dtype();

  // render the ast
  // first, count inputs
  std::vector<std::shared_ptr<AstLoadExpr>> leafs = get_leafs(ast);
  size_t n_inputs = leafs.size();
  std::string signature_str;
  for (size_t i = 0; i < n_inputs; i++) {
    AstLoadExpr inp = *leafs[i].get();
    signature_str +=
        "const" + dtype_to_string(inp.dtype) + " *in" + std::to_string(i);
  }

  std::vector<Tensor> inputs;
  for (size_t i = 0; i < n_inputs; i++) {
    inputs.push_back(std::move(*memo[leafs[i]].get()));
  }
  CompiledPrimitive compiled("kernel", store);
  out.ad_node().set_primitive(std::make_shared<CompiledPrimitive>(compiled));
  out.ad_node().set_children(inputs);
}
} // namespace pg