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
  if (is<Max>(prim)) {
    auto expr = std::make_shared<AstBinaryExpr>();
    expr->op = AstBinaryOp::Max;
    expr->lhs = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->rhs = get_ast_expr(curr.ad_node().children()[1], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  if (is<Gt>(prim)) {
    auto expr = std::make_shared<AstBinaryExpr>();
    expr->op = AstBinaryOp::Gt;
    expr->lhs = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->rhs = get_ast_expr(curr.ad_node().children()[1], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  if (is<Lt>(prim)) {
    auto expr = std::make_shared<AstBinaryExpr>();
    expr->op = AstBinaryOp::Lt;
    expr->lhs = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->rhs = get_ast_expr(curr.ad_node().children()[1], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  if (is<Eq>(prim)) {
    auto expr = std::make_shared<AstBinaryExpr>();
    expr->op = AstBinaryOp::Eq;
    expr->lhs = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->rhs = get_ast_expr(curr.ad_node().children()[1], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  if (is<Where>(prim)) {
    auto expr = std::make_shared<AstTernaryExpr>();
    expr->op = AstTernaryOp::Where;
    expr->first = get_ast_expr(curr.ad_node().children()[0], memo);
    expr->second = get_ast_expr(curr.ad_node().children()[1], memo);
    expr->third = get_ast_expr(curr.ad_node().children()[2], memo);
    expr->dtype = curr.dtype();
    return expr;
  }
  if (is<Fill>(prim)) {
    auto expr = std::make_shared<AstConstExpr>();
    expr->dtype = curr.dtype();
    expr->val = dynamic_cast<Fill *>(prim.get())->value();
    return expr;
  }
  if (is<Permute>(prim)) {
    auto expr = std::make_shared<AstPermuteOp>();
    // with permute, we must use a load as input
    auto load = std::make_shared<AstLoadExpr>();
    expr->child = load;
    // now since permute has only one child get from curr
    auto child_t = curr.ad_node().children()[0];
    // and stpre in load
    load->name = "in" + std::to_string(child_t.id);
    load->dtype = child_t.dtype();
    load->shape = child_t.shape();
    memo[load] = std::make_shared<Tensor>(child_t);
  }
  if (is<BroadcastTo>(prim)) {
    auto expr = std::make_shared<AstBroadcastOp>();
    // with broadcast, we must use a load as input
    auto load = std::make_shared<AstLoadExpr>();
    expr->child = load;
    // now since broadcast has only one child get from curr
    auto child_t = curr.ad_node().children()[0];
    // and stpre in load
    load->name = "in" + std::to_string(child_t.id);
    load->dtype = child_t.dtype();
    load->shape = child_t.shape();
    memo[load] = std::make_shared<Tensor>(child_t);
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
    if (std::dynamic_pointer_cast<AstTernaryExpr>(node)) {
      auto ternary = std::dynamic_pointer_cast<AstTernaryExpr>(node);
      recurse(ternary->first);
      recurse(ternary->second);
      recurse(ternary->third);
    }
    if (std::dynamic_pointer_cast<AstPermuteOp>(node)) {
      auto permute = std::dynamic_pointer_cast<AstPermuteOp>(node);
      return recurse(permute->child);
    }
    if (std::dynamic_pointer_cast<AstConstExpr>(node)) {
      return;
    }
    if (std::dynamic_pointer_cast<AstBroadcastOp>(node)) {
      auto broadcast = std::dynamic_pointer_cast<AstBroadcastOp>(node);
      return recurse(broadcast->child);
    }
  };
  recurse(node);
  return leafs;
}

int get_depth(std::shared_ptr<AstExpr> node) {
  using recurse_lambda_t = std::function<int(std::shared_ptr<AstExpr>)>;
  recurse_lambda_t recurse = [&](std::shared_ptr<AstExpr> node) {
    if (std::dynamic_pointer_cast<AstLoadExpr>(node)) {
      return 1;
    }
    if (std::dynamic_pointer_cast<AstUnaryExpr>(node)) {
      auto unary = std::dynamic_pointer_cast<AstUnaryExpr>(node);
      return 1 + recurse(unary->child);
    }
    if (std::dynamic_pointer_cast<AstBinaryExpr>(node)) {
      auto binary = std::dynamic_pointer_cast<AstBinaryExpr>(node);
      return 1 + std::max(recurse(binary->lhs), recurse(binary->rhs));
    }
    if (std::dynamic_pointer_cast<AstStoreExpr>(node)) {
      auto store = std::dynamic_pointer_cast<AstStoreExpr>(node);
      return recurse(store->value);
    }
    if (std::dynamic_pointer_cast<AstTernaryExpr>(node)) {
      auto ternary = std::dynamic_pointer_cast<AstTernaryExpr>(node);
      return 1 + std::max({recurse(ternary->first), recurse(ternary->second),
                           recurse(ternary->third)});
    }
    if (std::dynamic_pointer_cast<AstPermuteOp>(node)) {
      auto permute = std::dynamic_pointer_cast<AstPermuteOp>(node);
      return 1 + recurse(permute->child);
    }
    if (std::dynamic_pointer_cast<AstConstExpr>(node)) {
      return 1;
    }
    if (std::dynamic_pointer_cast<AstBroadcastOp>(node)) {
      auto broadcast = std::dynamic_pointer_cast<AstBroadcastOp>(node);
      return 1 + recurse(broadcast->child);
    }
  };
  return recurse(node);
}

bool fuse(Tensor &out) {
  std::map<std::shared_ptr<AstLoadExpr>, std::shared_ptr<Tensor>> memo;
  std::shared_ptr<AstExpr> ast = get_ast_expr(out, memo);
  // if ast is a load, it means we did not really do anything
  if (std::dynamic_pointer_cast<AstLoadExpr>(ast)) {
    return false;
  }
  // if depth of the ast is 1, we still specialize shapes and strides

  // Add a store operation after ast
  std::shared_ptr<AstStoreExpr> store = std::make_shared<AstStoreExpr>();
  store->name = "out";
  store->value = ast;
  store->dtype = out.dtype();

  // render the ast
  // first, count inputs
  std::vector<std::shared_ptr<AstLoadExpr>> leafs = get_leafs(ast);
  size_t n_inputs = leafs.size();
  if (n_inputs == 0) {
    return false;
  }
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

  return true;
}
} // namespace pg