#pragma once
#include "common/view_helpers.hpp"
#include "shape.hpp"
#include "tensor.hpp"
#include <stdexcept>
#include <vector>

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

#define DEFINE_INFER_OUTPUT_SHAPES                                             \
  std::vector<shape_t> infer_output_shapes(const std::vector<Tensor> &inputs)  \
      override;

#define DEFINE_INFER_OUTPUT_DTYPES                                             \
  std::vector<DType> infer_output_dtypes(const std::vector<Tensor> &inputs)    \
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
  virtual std::vector<shape_t>
  infer_output_shapes(const std::vector<Tensor> &inputs);

  virtual std::string str() { return "ADPrimitive"; }
  virtual std::vector<DType>
  infer_output_dtypes(const std::vector<Tensor> &inputs) {
    return {inputs[0].dtype()};
  }
};

class Log : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Log)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Exp : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Exp)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Add : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Add)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Mul : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Mul)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Sub : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Sub)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Div : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Div)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Gt : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Gt)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Lt : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Lt)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Eq : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Eq)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Neq : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Neq)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Ge : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Ge)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Le : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Le)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Pow : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Pow)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Max : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Max)
  DEFINE_INFER_OUTPUT_SHAPES
};

class Where : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Where)
  DEFINE_INFER_OUTPUT_SHAPES
};

// REDUCE
class Reduce : public ADPrimitive {
protected:
  axes_t _axes;
  bool _keepdims;

  const shape_t _reduce_single_shape_assuming_keepdims(View &input_view,
                                                       axis_t single_axis) {
    shape_t shape = shape_t(input_view.shape()); // copy the shape
    if (single_axis < 0) {
      single_axis = shape.size() + single_axis;
    }
    PG_CHECK_ARG(single_axis < shape.size(), "axis out of bounds, got ",
                 single_axis, " for shape ", vec_to_string(shape));
    shape[single_axis] = 1;
    return shape;
  }

public:
  DEFINE_STR_NAME(Reduce)
  explicit Reduce(axes_t axes, bool keepdims)
      : _axes(axes), _keepdims(keepdims) {}
};

class Sum : public Reduce {
  using Reduce::Reduce;

public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Sum)
  DEFINE_INFER_OUTPUT_SHAPES
};

class MaxReduce : public Reduce {
  using Reduce::Reduce;

public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(MaxReduce)
  DEFINE_BACKWARD
  DEFINE_INFER_OUTPUT_SHAPES
};

class Mean : public Reduce {
  using Reduce::Reduce;

public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Mean)
  DEFINE_BACKWARD
  DEFINE_INFER_OUTPUT_SHAPES
};

class BroadcastTo : public ADPrimitive {
protected:
  shape_t _shape_to;
  axes_t _broadcasted_axes;
  axes_t _created_axes;

public:
  explicit BroadcastTo(shape_t shape_to) : _shape_to(shape_to) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Broadcast)
  DEFINE_BACKWARD
  DEFINE_INFER_OUTPUT_SHAPES
};

class Squeeze : public ADPrimitive {
protected:
  axes_t _axes;

public:
  explicit Squeeze(axes_t axes) : _axes(axes) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Squeeze)
  DEFINE_BACKWARD
  DEFINE_INFER_OUTPUT_SHAPES
};

class Unsqueeze : public ADPrimitive {
protected:
  axes_t _axes;

public:
  explicit Unsqueeze(axes_t axes) : _axes(axes) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Unsqueeze)
  DEFINE_BACKWARD
  DEFINE_INFER_OUTPUT_SHAPES
};

class Permute : public ADPrimitive {
protected:
  axes_t _axes;

public:
  explicit Permute(axes_t axes) : _axes(axes) {
    PG_CHECK_ARG(axes.size() > 0, "Permute expects at least one axis");
  }
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Permute)
  DEFINE_BACKWARD
  DEFINE_INFER_OUTPUT_SHAPES
};

class MatMul : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(MatMul)
  DEFINE_BACKWARD
  DEFINE_INFER_OUTPUT_SHAPES
};

class Im2Col : public ADPrimitive {
protected:
  shape_t _kernel_shape;
  shape_t _strides;
  shape_t _padding;
  shape_t _dilation;

public:
  explicit Im2Col(shape_t kernel_shape, shape_t strides, shape_t padding,
                  shape_t dilation)
      : _kernel_shape(kernel_shape), _strides(strides), _padding(padding),
        _dilation(dilation) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Im2Col)
  DEFINE_INFER_OUTPUT_SHAPES
  DEFINE_BACKWARD
};

class Col2Im : public ADPrimitive {
protected:
  shape_t _output_shape;
  shape_t _kernel_shape;
  shape_t _strides;
  shape_t _padding;
  shape_t _dilation;

public:
  explicit Col2Im(shape_t output_shape, shape_t kernel_shape, shape_t strides,
                  shape_t padding, shape_t dilation)
      : _output_shape(output_shape), _kernel_shape(kernel_shape),
        _strides(strides), _padding(padding), _dilation(dilation) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Col2Im)
  DEFINE_INFER_OUTPUT_SHAPES
  DEFINE_BACKWARD
};

class Reshape : public ADPrimitive {
protected:
  axes_t _shape_to;

public:
  explicit Reshape(axes_t shape_to) : _shape_to(shape_to) {}
  DEFINE_DISPATCH_CUDA
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Reshape)
  DEFINE_INFER_OUTPUT_SHAPES
  DEFINE_BACKWARD
};

// SLICING

// Of the form [start:stop:step]
struct SelectWithSlice {
  int start;
  int stop;
  int step;
  SelectWithSlice(int start, int stop, int step)
      : start(start), stop(stop), step(step) {}
  SelectWithSlice(int start, int stop) : start(start), stop(stop), step(1) {}
  SelectWithSlice(int start) : start(start), stop(-1), step(1) {}
};

/*
Of the form
t = Tensor([1, 2, 3, 4, 5])
tidx = Tensor([0, 2, 4])
t[tidx] -> Tensor([1, 3, 5])
*/
// It is actually just a placeholder. Tensors will be passed as inputs lazily
// This just means 'expect a tensor here'
struct SelectWithTensor {};

/*
Of the form
t = Tensor([1, 2, 3, 4, 5])
tidx = 2
t[tidx] -> Tensor([3])
*/
struct SelectWithSingleIdx {
  long index;
  SelectWithSingleIdx(long index) : index(index) {}
};

/*
Keeps the dimension
*/
struct SelectKeepDim {
  SelectKeepDim() {}
};

using select_item_t = std::variant<SelectWithSlice, SelectWithTensor,
                                   SelectWithSingleIdx, SelectKeepDim>;
using select_t = std::vector<select_item_t>;
class Select : public ADPrimitive {
protected:
  select_t _items;

public:
  explicit Select(select_t items) : _items(items) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_INFER_OUTPUT_SHAPES
  DEFINE_STR_NAME(Select)
  DEFINE_BACKWARD
};

// 1 inp is dst, 2nd is src, rest are indices
// Output will be a copy of dst with src values at the indices
class AssignAt : public ADPrimitive {
protected:
  select_t _items;

public:
  explicit AssignAt(select_t items) : _items(items) {}
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(AssignAt)
  DEFINE_INFER_OUTPUT_SHAPES
  DEFINE_BACKWARD
  DEFINE_DISPATCH_CUDA
};

class AsContiguous : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(AsContiguous)
  DEFINE_INFER_OUTPUT_SHAPES
  DEFINE_BACKWARD
};

} // namespace pg