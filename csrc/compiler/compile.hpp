#pragma once
#include "pattern_matcher.hpp"
#include "scheduler.hpp"
#include "utils.hpp"

namespace pg {
namespace pm = pattern_matcher;
static bool is_broadcast(ADPrimitive &primitive) {
  return dynamic_cast<BroadcastTo *>(&primitive) != nullptr;
}

static bool is_broadcast(Tensor &tensor) {
  return is_broadcast(*tensor.ad_node()->primitive().get());
}

static BroadcastTo &get_broadcast(Tensor &tensor) {
  return dynamic_cast<BroadcastTo &>(*tensor.ad_node()->primitive().get());
}

static void remove_useless_broadcast(Tensor &out, std::set<int> &visited) {
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  for (Tensor &node : out.ad_node()->children()) {
    visited.insert(out.id);
    if (is_broadcast(node)) {
      auto &broadcast = get_broadcast(node);
      // useless broadcast
      if (broadcast.shape_to() == node.children()[0].shape()) {
        Tensor &child = node.ad_node()->children()[0];
        // connect out with the child
        out.ad_node()->replace_child(node, child);
        continue;
      }
    }
  }

  // now, recursively call remove_useless_broadcast for each children
  for (Tensor &node : out.ad_node()->children()) {
    remove_useless_broadcast(node, visited);
  }
}

// remove useless astype
static void remove_useless_astype(Tensor &out, std::set<int> &visited) {
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    if (node.ad_node()->primitive()->str() == "AsType") {
      auto &astype = dynamic_cast<AsType &>(*node.ad_node()->primitive().get());
      if (astype.dtype_to() == node.children()[0].dtype()) {
        Tensor &child = node.ad_node()->children()[0];
        out.ad_node()->replace_child(node, child);
      }
    }
  }

  for (Tensor &node : out.ad_node()->children()) {
    remove_useless_astype(node, visited);
  }
}

static void rec_schedule(Tensor &root, Tensor &out, std::set<int> &visited,
                         std::vector<Tensor> &allouts) {
  // get a map of tensor -> tensors that have that tensor as a child (depend on
  // it)
  std::unordered_map<int, std::set<int>>
      dependents; // key: tensor id, value: vector of tensor ids

  std::set<int> _visited;
  using recurse_t = std::function<void(Tensor &)>;
  recurse_t recurse = [&](Tensor &node) {
    if (_visited.find(node.id) != _visited.end()) {
      return;
    }
    _visited.insert(node.id);
    for (Tensor child : node.ad_node()->children()) {
      dependents[child.id].insert(node.id);
      recurse(child);
    }
  };
  recurse(root);
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  visited.insert(out.id);

  schedule(out, root, dependents);
  for (int i = 0; i < out.ad_node()->children().size(); i++) {
    if (i < out.ad_node()->children().size()) {
      rec_schedule(root, out.ad_node()->children()[i], visited, allouts);
    }
  }
}

#define RETURN_FALSE_IF_PRIM_IS_NOT(tensor, prim)                              \
  if (!(tensor.ad_node()->primitive()->str() == prim)) {                       \
    return std::nullopt;                                                       \
  }

template <typename T> using Maybe = std::optional<T>;

struct Conv2dPatternMatcherResult {
  Tensor input;
  Tensor filter;
  shape_t stride;
  shape_t dilation;
  shape_t padding;
};

Maybe<Conv2dPatternMatcherResult> conv2d_pattern_matcher(Tensor &out) {
  RETURN_FALSE_IF_PRIM_IS_NOT(out, "Col2Im");
  auto &foldchild = out.ad_node()->children()[0];
  // In conv, the folded output has 2 children: processed input and kernel
  RETURN_FALSE_IF_PRIM_IS_NOT(foldchild, "MatMul");
  auto &unfolded_input = foldchild.ad_node()->children()[1];
  auto &filter_permuted0 = foldchild.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(filter_permuted0,
                              "Broadcast"); // broadcasted filter

  auto &filter_permuted1 =
      filter_permuted0.ad_node()->children()[0]; // the actual permuted filter
  RETURN_FALSE_IF_PRIM_IS_NOT(filter_permuted1, "Permute");
  // now try to get filter
  RETURN_FALSE_IF_PRIM_IS_NOT(filter_permuted1.ad_node()->children()[0],
                              "Permute");
  auto &filter_permuted2 = filter_permuted1.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(filter_permuted2.ad_node()->children()[0],
                              "Reshape");
  auto &filter =
      filter_permuted2.ad_node()->children()[0].ad_node()->children()[0];
  // now try to get input
  RETURN_FALSE_IF_PRIM_IS_NOT(unfolded_input, "Im2Col");

  Im2Col &im2col =
      dynamic_cast<Im2Col &>(*unfolded_input.ad_node()->primitive().get());
  auto input = unfolded_input.ad_node()->children()[0];
  // if input primitive is assign_at, it is padding!
  shape_t padding = {0, 0};
  /* PADDING IS IMPLEMENTED LIKE
    out = pg.fill(
        new_shape,
        x.dtype,
        constant,
        x.device,
    )
    slices = [slice(int(pad[0]), int(-pad[1])) for pad in padpairs]

    for i, _slice in enumerate(slices):
        if _slice.start == 0 and _slice.stop == 0:
            slices[i] = slice(None, None, None)  # same as a[:]

    slices = tuple(slices)

    out = pg.assign_at(out, x, slices)*/
  // try to fuse padding
  if (input.ad_node()->primitive()->str() == "AssignAt" &&
      input.ad_node()->children().size() == 2) {
    auto &assign_at_dest = input.ad_node()->children()[0];
    auto &assign_at_src = input.ad_node()->children()[1];
    try {
      auto starts_with = [](const std::string &str, const std::string &prefix) {
        return str.rfind(prefix, 0) == 0;
      };
      if (starts_with(assign_at_dest.ad_node()->primitive()->str(), "Fill")) {

        PG_CHECK_RUNTIME(assign_at_src.shape().size() == 4,
                         "AssignAt src should be 4D, got " +
                             std::to_string(assign_at_src.shape().size()));
        PG_CHECK_RUNTIME(assign_at_dest.shape().size() == 4,
                         "AssignAt dest should be 4D, got " +
                             assign_at_dest.str());
        // now padding is the difference between the shapes in the last 2
        // dimensions (H, W)
        // TODO -- WILL NOT WORK WITH ASYMMETRIC PADDING
        padding = {(assign_at_dest.shape()[2] - assign_at_src.shape()[2]) / 2,
                   (assign_at_dest.shape()[3] - assign_at_src.shape()[3]) / 2};
        input = assign_at_src;
      }
    } catch (std::exception &e) {
    }
  }

  return Conv2dPatternMatcherResult{input, filter, im2col.strides(),
                                    im2col.dilation(), padding};
}

class Conv2dVjpWeightMatcherResult {
public:
  Tensor out_grad;
  Tensor input;
  shape_t stride;
  shape_t dilation;
  shape_t padding;
};

Maybe<Conv2dVjpWeightMatcherResult>
conv2d_vjp_weight_pattern_matcher(Tensor &out) {
  // use pattern matcher
  std::shared_ptr<Tensor> input_ptr = std::make_shared<Tensor>();
  std::shared_ptr<Tensor> out_grad_ptr = std::make_shared<Tensor>();
  std::shared_ptr<Tensor> im2col_ptr = std::make_shared<Tensor>();
  auto p = pm::Reshape(pm::Permute(pm::Permute(pm::Sum(
      pm::MatMul(pm::Im2Col(pm::Input(out_grad_ptr)),
                 pm::Permute(pm::Im2Col(pm::Input(input_ptr), im2col_ptr)))))));
  if (!p->match(out)) {
    return std::nullopt;
  }
  auto *im2col =
      dynamic_cast<Im2Col *>(im2col_ptr->ad_node()->primitive().get());
  PG_CHECK_RUNTIME(im2col != nullptr,
                   "Im2Col not found, str is " +
                       im2col_ptr->ad_node()->primitive()->str());
  return Conv2dVjpWeightMatcherResult{
      *out_grad_ptr, *input_ptr, im2col->strides(), im2col->dilation(), {0, 0}};

  /*RETURN_FALSE_IF_PRIM_IS_NOT(out, "Reshape");
  auto &out1 = out.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(out1, "Permute");
  auto &out2 = out1.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(out2, "Permute");
  auto &out3 = out2.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(out3, "Sum");
  auto &out4 = out3.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(out4, "MatMul");
  auto &out_grad0 = out4.ad_node()->children()[0];
  auto &input = out4.ad_node()->children()[1];
  // OUT GRAD
  RETURN_FALSE_IF_PRIM_IS_NOT(out_grad0, "Im2Col");
  auto &out_grad = out_grad0.ad_node()->children()[0];

  // INPUT
  RETURN_FALSE_IF_PRIM_IS_NOT(input, "Permute");
  auto &input0 = input.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(input0, "Im2Col");
  auto &input1 = input0.ad_node()->children()[0];
  auto &im2col = dynamic_cast<Im2Col &>(*input0.ad_node()->primitive().get());

  return Conv2dVjpWeightMatcherResult{
      out_grad, input1, im2col.strides(), im2col.dilation(), {0, 0}};
      */
}

class Conv2dVjpInputMatcherResult {
public:
  Tensor out_grad;
  Tensor filter;
  shape_t stride;
  shape_t dilation;
  shape_t padding;
};

Maybe<Conv2dVjpInputMatcherResult>
conv2d_vjp_input_pattern_matcher(Tensor &out) {
  RETURN_FALSE_IF_PRIM_IS_NOT(out, "Col2Im");
  auto &out1 = out.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(out1, "MatMul");
  auto &filter0 = out1.ad_node()->children()[0];
  auto &out_grad0 = out1.ad_node()->children()[1];
  // OUT GRAD
  RETURN_FALSE_IF_PRIM_IS_NOT(out_grad0, "Im2Col");
  auto &out_grad = out_grad0.ad_node()->children()[0];
  // FILTER
  RETURN_FALSE_IF_PRIM_IS_NOT(filter0, "Permute");
  auto &filter1 = filter0.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(filter1, "Broadcast");
  auto &filter2 = filter1.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(filter2, "Permute");
  auto &filter = filter2.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(filter, "Permute");
  auto &filter3 = filter.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(filter3, "Reshape");
  auto &filter4 = filter3.ad_node()->children()[0];

  auto &col2im = dynamic_cast<Col2Im &>(*out.ad_node()->primitive().get());
  return Conv2dVjpInputMatcherResult{
      out_grad, filter4, col2im.strides(), col2im.dilation(), {0, 0}};
}

bool try_convert_conv2d_vjp_input(Tensor &out) {
  auto maybe_result = conv2d_vjp_input_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  auto &input = result.filter;
  auto &out_grad = result.out_grad;
  auto kernel_size = shape_t{input.shape()[2], input.shape()[3]};
  out.ad_node()->set_primitive(std::make_shared<CudnnConv2dVjpInput>(
      result.stride, result.dilation, kernel_size, result.padding));
  out.ad_node()->set_children({input, out_grad});
  return true;
}

bool try_convert_conv2d_vjp_weight(Tensor &out) {
  auto maybe_result = conv2d_vjp_weight_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  auto &input = result.input;
  auto &out_grad = result.out_grad;
  auto kernel_size = shape_t{out_grad.shape()[2], out_grad.shape()[3]};
  out.ad_node()->set_primitive(std::make_shared<CudnnConv2dVjpWeight>(
      result.stride, result.dilation, kernel_size, result.padding));
  out.ad_node()->set_children({input, out_grad});
  return true;
}

void recursive_conv2d_vjp_weight(Tensor &out, std::set<int> &visited) {
  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  try_convert_conv2d_vjp_weight(out);
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_conv2d_vjp_weight(node, visited);
  }
}
struct Pooling2dPatternMatcherResult {
  Tensor input;
  std::string reduce_type;
  shape_t stride;
  shape_t kernel_size;
};

Maybe<Pooling2dPatternMatcherResult> pooling2d_pattern_matcher(Tensor &out) {

  auto &unreshaped = out;
  // now it can be MaxReduce or Mean
  std::string reduce_type = "";
  if (unreshaped.ad_node()->primitive()->str() == "MaxReduce") {
    reduce_type = "MaxReduce";
  } else if (unreshaped.ad_node()->primitive()->str() == "Mean") {
    reduce_type = "Mean";
  } else {
    return std::nullopt;
  }
  auto &unreduced = unreshaped.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(unreduced, "Reshape");
  auto &inputreshaped = unreduced.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(inputreshaped, "Im2Col");
  auto &input = inputreshaped.ad_node()->children()[0];
  auto &im2col =
      dynamic_cast<Im2Col &>(*inputreshaped.ad_node()->primitive().get());
  auto stride = im2col.strides();
  auto kernel_size = im2col.kernel_shape();
  return Pooling2dPatternMatcherResult{input, reduce_type, stride, kernel_size};
}

struct MaxPooling2dBackwardPatternMatcherResult {
  Tensor forward_out;
  Tensor input;
  Tensor grad_out;
  shape_t stride;
  shape_t kernel_size;
};

Maybe<MaxPooling2dBackwardPatternMatcherResult>
max_pooling2d_backward_pattern_matcher(Tensor &out) {
  RETURN_FALSE_IF_PRIM_IS_NOT(out, "Col2Im");
  auto &i0 = out.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(i0, "Reshape");
  auto &i1 = i0.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(i1, "Where");
  auto &i2 = i1.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(i2, "Eq");
  auto &i3 = i2.ad_node()->children()[1];
  RETURN_FALSE_IF_PRIM_IS_NOT(i3, "Broadcast");
  auto &i4 = i3.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(i4, "Unsqueeze");
  auto &i5 = i4.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(i5, "CudnnPooling2D<MaxReduce>");
  auto &fw_out = i5;
  auto &input = i5.ad_node()->children()[0];
  auto &pool = dynamic_cast<CudnnPooling2D &>(*i5.ad_node()->primitive().get());
  auto &i7 = i1.ad_node()->children()[1];
  RETURN_FALSE_IF_PRIM_IS_NOT(i7, "Broadcast");
  auto &i8 = i7.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(i8, "Unsqueeze");
  auto &i9 = i8.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(i9, "Reshape");
  auto &grad_out = i9.ad_node()->children()[0];

  return MaxPooling2dBackwardPatternMatcherResult{
      fw_out, input, grad_out, pool.strides, pool.kernel_shape};
}

bool try_convert_max_pooling2d_backward(Tensor &out) {
  auto maybe_result = max_pooling2d_backward_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  out.ad_node()->set_primitive(std::make_shared<CudnnPooling2DVjp>(
      result.kernel_size, result.stride, "Max"));
  out.ad_node()->set_children(
      {result.forward_out, result.input, result.grad_out});
  return true;
}

void recursive_max_pooling2d_backward(Tensor &out, std::set<int> &visited) {
  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  try_convert_max_pooling2d_backward(out);
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_max_pooling2d_backward(node, visited);
  }
}

class LocalResponseNormalizationPatternMatcherResult {
public:
  Tensor input;
  int size;
  float alpha;
  float beta;
  float k;
};

// define a 'tree' to pattern match

Maybe<LocalResponseNormalizationPatternMatcherResult>
local_response_normalization_pattern_matcher(Tensor &out) {
  RETURN_FALSE_IF_PRIM_IS_NOT(out, "Div");
  auto &div = out.ad_node()->children()[1];
  // -----------------
  RETURN_FALSE_IF_PRIM_IS_NOT(div, "Broadcast");
  auto &broadcast = div.ad_node()->children()[0];
  // -----------------
  RETURN_FALSE_IF_PRIM_IS_NOT(broadcast, "Pow");
  auto &pow0 = broadcast.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(pow0, "Add");

  auto &add0 = pow0.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(add0, "Mul");

  auto &mul0 = add0.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(mul0, "Mean");
  auto &mean = mul0.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(mean, "Pow");
  auto &pow2 = mean.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(pow2, "Reshape");
  auto &reshape = pow2.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(reshape, "Im2Col");
  auto &inputprv = reshape.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(inputprv, "AssignAt"); // this handles the padding
  auto &input = inputprv.ad_node()->children()[1];
  //   nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
  // now we have the input
  return LocalResponseNormalizationPatternMatcherResult{input, 5, 1e-4, 0.75,
                                                        2};
}

bool try_convert_local_response_normalization(Tensor &out) {
  auto maybe_result = local_response_normalization_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  out.ad_node()->set_primitive(std::make_shared<CudnnLRN>(
      result.size, result.alpha, result.beta, result.k));
  out.ad_node()->set_children({result.input});
  return true;
}

void recursive_local_response_normalization(Tensor &out,
                                            std::set<int> &visited) {
  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  try_convert_local_response_normalization(out);
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_local_response_normalization(node, visited);
  }
}

class LRNVjpInputPatternMatcherResult {
public:
  Tensor out_grad;
  Tensor input;
  Tensor out; // the output of the forward
  int size;
  float alpha;
  float beta;
  float k;
};

Maybe<LRNVjpInputPatternMatcherResult>
lrn_vjp_input_pattern_matcher(Tensor &out) {
  RETURN_FALSE_IF_PRIM_IS_NOT(out, "Add");
  RETURN_FALSE_IF_PRIM_IS_NOT(out.ad_node()->children()[1], "CudnnLRN");
  auto &x0 = out.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x0, "Select");
  auto &x1 = x0.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x1, "Col2Im");
  auto &x2 = x1.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x2, "Reshape");
  auto &x3 = x2.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x3, "Mul");
  auto &x4 = x3.ad_node()->children()[1];
  RETURN_FALSE_IF_PRIM_IS_NOT(x4, "Broadcast");
  auto &x5 = x4.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x5, "Div");
  auto &x6 = x5.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x6, "Broadcast");
  auto &x7 = x6.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x7, "Mul");
  auto &x8 = x7.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x8, "Mul");
  auto &x9 = x8.ad_node()->children()[1];
  RETURN_FALSE_IF_PRIM_IS_NOT(x9, "Sum");
  auto &x10 = x9.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x10, "Div");
  auto &x11 = x10.ad_node()->children()[0];
  RETURN_FALSE_IF_PRIM_IS_NOT(x11, "Mul");
  auto &x12 = x11.ad_node()->children()[1];
  RETURN_FALSE_IF_PRIM_IS_NOT(x12, "Mul");
  auto &x13 = x12.ad_node()->children()[0];
  auto grad_out = x13;
  // now get input from out.ad_node()->children()[1]
  auto &input = out.ad_node()->children()[1].ad_node()->children()[0];
  auto &lrn_out = out.ad_node()->children()[1];
  auto &lrn = dynamic_cast<CudnnLRN &>(
      *out.ad_node()->children()[1].ad_node()->primitive().get());

  return LRNVjpInputPatternMatcherResult{grad_out, input, lrn_out, 5,
                                         1e-4,     0.75,  2};
}

bool try_convert_lrn_vjp_input(Tensor &out) {
  auto maybe_result = lrn_vjp_input_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  out.ad_node()->set_primitive(std::make_shared<CudnnLRNVjpInput>(
      result.size, result.alpha, result.beta, result.k));
  out.ad_node()->set_children({result.out, result.out_grad, result.input});
  return true;
}

void recursive_lrn_vjp_input(Tensor &out, std::set<int> &visited) {
  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  try_convert_lrn_vjp_input(out);
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_lrn_vjp_input(node, visited);
  }
}

bool try_convert_conv2d(Tensor &out) {
  auto maybe_result = conv2d_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  auto &input = result.input;
  auto &filter = result.filter;
  auto kernel_size = shape_t{filter.shape()[2], filter.shape()[3]};
  out.ad_node()->set_primitive(std::make_shared<CudnnConv2D>(
      result.stride, result.dilation, kernel_size, result.padding));
  out.ad_node()->set_children({input, filter});
  return true;
}

bool try_convert_pooling2d(Tensor &out) {
  auto maybe_result = pooling2d_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  auto &input = result.input;
  out.ad_node()->set_primitive(std::make_shared<CudnnPooling2D>(
      result.kernel_size, result.stride, result.reduce_type));
  out.ad_node()->set_children({input});
  return true;
}

void recursive_conv2d(Tensor &out, std::set<int> &visited) {
  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  try_convert_conv2d(out);
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_conv2d(node, visited);
  }
}

void recursive_conv2d_vjp_input(Tensor &out, std::set<int> &visited) {
  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  try_convert_conv2d_vjp_input(out);
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_conv2d_vjp_input(node, visited);
  }
}

void recursive_pooling2d(Tensor &out, std::set<int> &visited) {
  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  try_convert_pooling2d(out);
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_pooling2d(node, visited);
  }
}

class FusedLinearBiasReLUPatternMatcherResult {
public:
  Tensor input;
  Tensor weight;
  Tensor bias;
};

Maybe<FusedLinearBiasReLUPatternMatcherResult>
fused_linear_pattern_matcher(Tensor &out) {

  std::shared_ptr<Tensor> input_ptr = std::make_shared<Tensor>();
  std::shared_ptr<Tensor> weight_ptr = std::make_shared<Tensor>();
  std::shared_ptr<Tensor> bias_ptr = std::make_shared<Tensor>();

  pm::Pattern p =
      pm::Add(pm::MatMul(pm::Input(input_ptr), pm::Input(weight_ptr)),
              pm::Broadcast(pm::Input(bias_ptr)));

  if (!p->match(out)) {
    return std::nullopt;
  }

  // m must be multiple of 16, n and k of 8
  auto m = weight_ptr->shape()[0];
  auto n = weight_ptr->shape()[1];
  auto k = input_ptr->shape()[1];
  if (m % 16 != 0 || n % 8 != 0 || k % 8 != 0) {
    return std::nullopt;
  }
  return FusedLinearBiasReLUPatternMatcherResult{*input_ptr, *weight_ptr,
                                                 *bias_ptr};
}

bool try_convert_fused_linear(Tensor &out) {
  auto maybe_result = fused_linear_pattern_matcher(out);
  if (!maybe_result.has_value()) {
    return false;
  }
  auto result = maybe_result.value();
  out.ad_node()->set_primitive(std::make_shared<FusedLinearBiasReLU>());
  out.ad_node()->set_children({result.input, result.weight, result.bias});
  return true;
}

void recursive_fused_linear(Tensor &out, std::set<int> &visited) {
  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  try_convert_fused_linear(out);
  visited.insert(out.id);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_fused_linear(node, visited);
  }
}

bool is_elwise(Tensor &out) {
  std::string str = out.ad_node()->primitive()->str();
  std::vector<std::string> supported = {"Exp", "Log", "Add", "Sub", "Mul"};
  return std::find(supported.begin(), supported.end(), str) != supported.end();
}

void hoist_broadcasts(Tensor &out, std::set<int> &visited) {
  // having a graph like
  // input -> elwise -> broadcast -> elwise into input -> broadcast -> elwise ->
  // elwise this allows for better fusion

  if (out.device() != device::CUDA) {
    return;
  }
  if (visited.find(out.id) != visited.end()) {
    return;
  }

  if (is_elwise(out)) {
    for (Tensor maybe_bc : out.ad_node()->children()) {
      if (is_broadcast(maybe_bc)) {
        // output <- elwise <- broadcast <- broadcast_child
        // into output <- elwise <- broadcast_child <- broadcast
        auto &broadcast_node = get_broadcast(maybe_bc);
        auto broadcast_child = maybe_bc.ad_node()->children()[0];
        if (!is_elwise(broadcast_child) ||
            broadcast_child.ad_node()->children().size() != 1) {
          continue;
        }
        auto bc_child_child = broadcast_child.ad_node()->children()[0];

        out.ad_node()->replace_child(maybe_bc, broadcast_child);
        broadcast_child.ad_node()->replace_child(bc_child_child, maybe_bc);
        maybe_bc.ad_node()->replace_child(broadcast_child, bc_child_child);

        auto maybe_bc_prim_ptr = maybe_bc.ad_node()->primitive();
        maybe_bc.copy_view_inplace(
            maybe_bc_prim_ptr->precompute(maybe_bc.ad_node()->children())[0]);

        // now precompute to propagate shapes
        auto bc_child_prim_ptr = broadcast_child.ad_node()->primitive();
        broadcast_child.copy_view_inplace(bc_child_prim_ptr->precompute(
            broadcast_child.ad_node()->children())[0]);

        auto out_prim_ptr = out.ad_node()->primitive();
        out.copy_view_inplace(
            out_prim_ptr->precompute(out.ad_node()->children())[0]);
      }
    }
  }

  visited.insert(out.id);

  for (Tensor &node : out.ad_node()->children()) {
    hoist_broadcasts(node, visited);
  }
}

// COMMON SUBEXPRESSION ELIMINATION
// hash a tensor
static std::string hash_tensor(Tensor &t, std::map<std::string, Tensor> &hash2ten, std::map<int, std::string> &ten2hash, int* nonelim_count) {
  // if visited, return
  if (ten2hash.find(t.id) != ten2hash.end()) {
    return ten2hash[t.id];
  }
  std::string hash = t.ad_node()->primitive()->str();
  // if we are using non-eliminable primitives, we need to add a random number
  std::array<std::string, 4> nonelim = {"ADPrimitive", "FromNumpy", "Compiled", "JitBoundary"};
  for (const auto& substr : nonelim) {
    if (hash.find(substr) != std::string::npos) {
      hash += "--" + std::to_string(*nonelim_count) + "--";
      (*nonelim_count)++;
      break;
    }
  }
  // add shape, stride, dtype to hash
  auto get_str = [](const std::vector<unsigned long long> &shape) {
    std::string str = "";
    for (auto &s : shape) {
      str += std::to_string(s) + ",";
    }
    return str;
  };

  auto get_str2 = [](const std::vector<long> &shape) {
    std::string str = "";
    for (auto &s : shape) {
      str += std::to_string(s) + ",";
    }
    return str;
  };

  hash += get_str(t.shape());
  hash += get_str2(t.strides());

  hash += int(t.dtype());


  for (Tensor &child : t.ad_node()->children()) {
    hash += hash_tensor(child, hash2ten, ten2hash, nonelim_count);
  }
  hash2ten[hash] = t;
  ten2hash[t.id] = hash;
  return hash;
}

void _common_subexpr_elim_recursive(Tensor &out, std::map<std::string, Tensor> &hash2ten, std::map<int, std::string> &ten2hash) {
  for (Tensor &child : out.ad_node()->children()) {
    out.ad_node()->replace_child(child, hash2ten[ten2hash[child.id]]);
  }
  for (Tensor &child : out.ad_node()->children()) {
    _common_subexpr_elim_recursive(child, hash2ten, ten2hash);
  }
}
void common_subexpr_elim(std::vector<Tensor> &outs) {
  // This automatically removes common subexpressions
  // Since for any 2 tensors, if they have the same hash, they are the same
  // Since it is a map, the contents of hashes[hash] will be canonical
  std::map<std::string, Tensor> hash2ten;
  std::map<int, std::string> ten2hash;
  int nonelim_count = 0;
  for (Tensor &out : outs) {
    hash_tensor(out, hash2ten, ten2hash, &nonelim_count);
  }
  for (Tensor &out : outs) {
    _common_subexpr_elim_recursive(out, hash2ten, ten2hash);
  }
}

static bool is_copy(ADPrimitive &primitive) {
  return dynamic_cast<Copy *>(&primitive) != nullptr;
}

static bool is_copy(Tensor &tensor) {
  return is_copy(*tensor.ad_node()->primitive().get());
}

static void remove_useless_copy(Tensor &out, std::set<int> &visited) {
  if (visited.find(out.id) != visited.end()) {
    return;
  }
  for (Tensor &node : out.ad_node()->children()) {
    visited.insert(out.id);
    if (is_copy(node)) {
        Tensor &child = node.ad_node()->children()[0];
        // connect out with the child
        out.ad_node()->replace_child(node, child);
        continue;
      }
  }
  

  for (Tensor &node : out.ad_node()->children()) {
    remove_useless_copy(node, visited);
  }
}


class CompileOptions {
public:
  bool remove_useless_copy = true;
  bool remove_useless_broadcast = true;
  bool remove_useless_astype = true;
  bool recursive_fused_linear = true;
  bool recursive_conv2d = true;
  bool recursive_pooling2d = true;
  bool recursive_conv2d_vjp_weight = true;
  bool recursive_conv2d_vjp_input = true;
  bool recursive_local_response_normalization = true;
  bool recursive_lrn_vjp_input = true;
  bool recursive_max_pooling2d_backward = true;
  bool hoist_broadcasts = true;
  bool common_subexpr_elim = true;
  bool fuser = true;
};

#define COMPILER_DBG 1
#define COMPILER_LOG(x)                                                        \
  if (COMPILER_DBG) {                                                          \
    std::cout << "[DEBUG]: " << x << "\n";                                     \
  }
static void _compile(std::vector<Tensor> &outs, CompileOptions options = CompileOptions()) {
  for (Tensor &out : outs) {
    COMPILER_LOG("compiling " << out.str());
    std::set<int> visited;
    if (options.remove_useless_copy) {
      visited.clear();
      remove_useless_copy(out, visited);
      COMPILER_LOG("removed useless copy");
    }
    if (options.remove_useless_broadcast) {
      visited.clear();
      remove_useless_broadcast(out, visited);
      COMPILER_LOG("removed useless broadcasts");
    }
    if (options.remove_useless_astype) {
      visited.clear();
      remove_useless_astype(out, visited);
      COMPILER_LOG("removed useless astype");
    }
    std::set<int> visited1;
    if (options.recursive_fused_linear) {
      recursive_fused_linear(out, visited1);
      COMPILER_LOG("fused linear");
    }
    visited.clear();
    if (options.recursive_conv2d) {
      recursive_conv2d(out, visited);
      COMPILER_LOG("conv2d");
    }
    visited.clear();
    if (options.recursive_pooling2d) {
      recursive_pooling2d(out, visited);
      COMPILER_LOG("pooling2d");
    }
    visited.clear();
    if (options.recursive_conv2d_vjp_weight) {
      recursive_conv2d_vjp_weight(out, visited);
      COMPILER_LOG("conv2d vjp weight");
    }
    visited.clear();
    if (options.recursive_conv2d_vjp_input) {
      recursive_conv2d_vjp_input(out, visited);
      COMPILER_LOG("conv2d vjp input");
    }
    visited.clear();
    if (options.recursive_local_response_normalization) {
      recursive_local_response_normalization(out, visited);
      COMPILER_LOG("local response normalization");
    }
    visited.clear();
    if (options.recursive_lrn_vjp_input) {
      recursive_lrn_vjp_input(out, visited);
      COMPILER_LOG("lrn vjp input");
    }
    visited.clear();
    if (options.recursive_max_pooling2d_backward) {
      recursive_max_pooling2d_backward(out, visited);
      COMPILER_LOG("max pooling2d backward");
    }
    visited.clear();
    if (options.hoist_broadcasts) {
      hoist_broadcasts(out, visited);
      COMPILER_LOG("hoist broadcasts");
    }
    std::set<int> visited2;
    if (options.fuser) {
      rec_schedule(out, out, visited2, outs);
      COMPILER_LOG("scheduled");
    }
  }
  if (options.common_subexpr_elim) {
    common_subexpr_elim(outs);
    COMPILER_LOG("common subexpr elim");
  }
}

static void compile(std::vector<Tensor> &outs, std::map<std::string, bool> options = {}) {
  CompileOptions compile_options;
  compile_options.remove_useless_copy = options.find("remove_useless_copy") != options.end() ? options["remove_useless_copy"] : true;
  compile_options.remove_useless_broadcast = options.find("remove_useless_broadcast") != options.end() ? options["remove_useless_broadcast"] : true;
  compile_options.remove_useless_astype = options.find("remove_useless_astype") != options.end() ? options["remove_useless_astype"] : true;
  compile_options.recursive_fused_linear = options.find("recursive_fused_linear") != options.end() ? options["recursive_fused_linear"] : true;
  compile_options.recursive_conv2d = options.find("recursive_conv2d") != options.end() ? options["recursive_conv2d"] : true;
  compile_options.recursive_pooling2d = options.find("recursive_pooling2d") != options.end() ? options["recursive_pooling2d"] : true;
  compile_options.recursive_conv2d_vjp_weight =
      options.find("recursive_conv2d_vjp_weight") != options.end() ? options["recursive_conv2d_vjp_weight"] : true;
  compile_options.recursive_conv2d_vjp_input =
      options.find("recursive_conv2d_vjp_input") != options.end() ? options["recursive_conv2d_vjp_input"] : true;
  compile_options.recursive_local_response_normalization =
      options.find("recursive_local_response_normalization") != options.end() ? options["recursive_local_response_normalization"] : true;
  compile_options.recursive_lrn_vjp_input = options.find("recursive_lrn_vjp_input") != options.end() ? options["recursive_lrn_vjp_input"] : true;
  compile_options.recursive_max_pooling2d_backward =
      options.find("recursive_max_pooling2d_backward") != options.end() ? options["recursive_max_pooling2d_backward"] : true;
  compile_options.hoist_broadcasts = options.find("hoist_broadcasts") != options.end() ? options["hoist_broadcasts"] : true;
  compile_options.common_subexpr_elim = options.find("common_subexpr_elim") != options.end() ? options["common_subexpr_elim"] : true;
  compile_options.fuser = options.find("fuser") != options.end() ? options["fuser"] : true;
  _compile(outs, compile_options);
}

} // namespace pg