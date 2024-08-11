#pragma once
#include "scheduler.hpp"

namespace pg {
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

static void rec_schedule(Tensor &root, Tensor &out, std::set<int> &visited) {
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
      rec_schedule(root, out.ad_node()->children()[i], visited);
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
  RETURN_FALSE_IF_PRIM_IS_NOT(out, "Reshape");
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

void recursive_conv2d_vjp_weight(Tensor &out) {
  if (out.device() != device::CUDA) {
    return;
  }
  try_convert_conv2d_vjp_weight(out);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_conv2d_vjp_weight(node);
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

void recursive_max_pooling2d_backward(Tensor &out) {
  if (out.device() != device::CUDA) {
    return;
  }
  try_convert_max_pooling2d_backward(out);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_max_pooling2d_backward(node);
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

void recursive_local_response_normalization(Tensor &out) {
  if (out.device() != device::CUDA) {
    return;
  }
  try_convert_local_response_normalization(out);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_local_response_normalization(node);
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

void recursive_lrn_vjp_input(Tensor &out) {
  if (out.device() != device::CUDA) {
    return;
  }
  try_convert_lrn_vjp_input(out);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_lrn_vjp_input(node);
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

void recursive_conv2d(Tensor &out) {
  if (out.device() != device::CUDA) {
    return;
  }
  try_convert_conv2d(out);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_conv2d(node);
  }
}

void recursive_conv2d_vjp_input(Tensor &out) {
  if (out.device() != device::CUDA) {
    return;
  }
  try_convert_conv2d_vjp_input(out);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_conv2d_vjp_input(node);
  }
}

void recursive_pooling2d(Tensor &out) {
  if (out.device() != device::CUDA) {
    return;
  }
  try_convert_pooling2d(out);
  for (Tensor &node : out.ad_node()->children()) {
    recursive_pooling2d(node);
  }
}

bool is_elwise(Tensor &out) {
  std::string str = out.ad_node()->primitive()->str();
  std::vector<std::string> supported = {"Exp", "Log", "Add", "Sub", "Mul"};
  return std::find(supported.begin(), supported.end(), str) != supported.end();
}

void hoist_broadcasts(Tensor &out) {
  // having a graph like
  // input -> elwise -> broadcast -> elwise into input -> broadcast -> elwise ->
  // elwise this allows for better fusion

  if (out.device() != device::CUDA) {
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

  for (Tensor &node : out.ad_node()->children()) {
    hoist_broadcasts(node);
  }
}
static void compile(std::vector<Tensor> &outs) {
  for (Tensor &out : outs) {
    std::set<int> visited;
    remove_useless_broadcast(out, visited);
    recursive_conv2d(out);
    recursive_pooling2d(out);
    recursive_conv2d_vjp_weight(out);
    recursive_conv2d_vjp_input(out);
    recursive_local_response_normalization(out);
    recursive_lrn_vjp_input(out);
    recursive_max_pooling2d_backward(out);
    hoist_broadcasts(out);
    std::set<int> visited2;
    rec_schedule(out, out, visited2);
  }
}
} // namespace pg