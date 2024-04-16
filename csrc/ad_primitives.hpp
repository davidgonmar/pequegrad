#pragma once
#include "shape.hpp"
#include "tensor.hpp"
#include <stdexcept>
#include <vector>
#include "common/view_helpers.hpp"


#define DEFINE_DISPATCH_CPU                                                    \
  void dispatch_cpu(const std::vector<Tensor> &inputs,                         \
                    std::vector<Tensor> &outputs) override;

#define DEFINE_DISPATCH_CUDA                                                   \
  void dispatch_cuda(const std::vector<Tensor> &inputs,                        \
                     std::vector<Tensor> &outputs) override;

#define DEFINE_BACKWARD                                                        \
  std::vector<Tensor> backward(const std::vector<Tensor> &primals,             \
                               const std::vector<Tensor> &tangents,            \
                               const std::vector<Tensor> &outputs) override;

#define DEFINE_OUTPUT_SHAPES                                                   \
  std::vector<shape_t> output_shapes(const std::vector<Tensor> &inputs)        \
      override;

#define DEFINE_STR_NAME(NAME)                                                  \
  std::string str() { return #NAME; }


namespace pg {
class ADPrimitive {
public:
  /**
   * Dispatch is responsible for allocating the memory, and populating
   * the `View` objects in the output tensors. The `View` objects have
   * the data ptrs, as well as strides/shape information and dtypes.
   */
  virtual void dispatch_cpu(const std::vector<Tensor> &inputs,
                            std::vector<Tensor> &outputs);
  virtual void dispatch_cuda(const std::vector<Tensor> &inputs,
                             std::vector<Tensor> &outputs);

  /**
   * The backward pass of the primitive. It does not really take care of the
   * computation, but uses other primitives to compute the gradients.
   * This allows for higher order derivatives.
   */
  virtual std::vector<Tensor> backward(const std::vector<Tensor> &primals,
                                       const std::vector<Tensor> &tangents,
                                       const std::vector<Tensor> &outputs);

  /** Get the output shapes of the primitive.
   * Strides are backend specific, so we don't need to worry about them here.
   */
  virtual std::vector<shape_t> output_shapes(const std::vector<Tensor> &inputs);

  virtual std::string str() {
    return "ADPrimitive";
  }
};


class Log : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Log)
};

class Add : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Add)
};

class Mul : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Mul)
};

class Sub : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Sub)
};

class Div : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Div)
};

class Gt : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Gt)
};

class Lt : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Lt)
};

class Eq : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Eq)
};

class Neq : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Neq)
};

class Ge : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Ge)
};

class Le : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Le)
};

class Pow : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Pow)
};

class Max : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Max)
};

// REDUCE
class Reduce : public ADPrimitive {
protected:
  axes_t _axes;
  bool _keepdims;

  const shape_t _reduce_single_shape_assuming_keepdims(View &input_view, axis_t single_axis) {
    shape_t shape = shape_t(input_view.shape()); // copy the shape
    if (single_axis < 0) {
      single_axis = shape.size() + single_axis;
    }
    PG_CHECK_ARG(single_axis < shape.size(), "axis out of bounds, got ", single_axis,
                 " for shape ", vec_to_string(shape));
    shape[single_axis] = 1;
    return shape;
  }

  const View _unreduce_for_backward(const shape_t original_shape, const View reduced_view) {
    return std::get<0>(view::broadcasted_to(reduced_view, original_shape)); // TODO -- when keepdims is false, we need to first expand dims
  }

public:
  DEFINE_STR_NAME(Reduce)
  explicit Reduce(axes_t axes, bool keepdims) : _axes(axes), _keepdims(keepdims) {}
};

class Sum : public Reduce {
using Reduce::Reduce;
public:
  DEFINE_DISPATCH_CPU
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Sum)
};

class MaxReduce : public Reduce {
using Reduce::Reduce;
public:
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(MaxReduce)
};

class Mean : public Reduce {
using Reduce::Reduce;
public:
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Mean)
};

class BroadcastTo : public ADPrimitive {
protected:
  shape_t _shape_to;
  axes_t _axes_to_reduce_in_bw; // will be populated when the op is run
public:
  explicit BroadcastTo(shape_t shape_to) : _shape_to(shape_to) {}
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Broadcast)
  DEFINE_BACKWARD
};


} // namespace pg