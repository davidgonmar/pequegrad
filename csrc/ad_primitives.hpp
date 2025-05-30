#pragma once
#include "shape.hpp"
#include "tensor.hpp"
#include <cudnn.h>
#include <optional>
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
#define DEFINE_STR_NAME(NAME)                                                  \
  std::string str() { return #NAME; }

#define DEFINE_PRECOMPUTE                                                      \
  std::vector<View> precompute(const std::vector<Tensor> &inputs) override;
namespace pg {
class ADPrimitive {
public:
  virtual bool eager() { return false; }
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

  virtual std::vector<View> precompute(const std::vector<Tensor> &inputs) {
    throw std::runtime_error("precompute not implemented for " + str());
  }

  virtual std::string str() { return "ADPrimitive"; }
};

class JitBoundary : public ADPrimitive {
public:
  DEFINE_STR_NAME(JitBoundary)
  explicit JitBoundary() {}
};

class CudnnConv2D : public ADPrimitive {
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(CudnnConv2D)
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  int workspace_size;
  cudnnConvolutionFwdAlgo_t algo;
  bool initialized = false;
  void *workspace = nullptr;

public:
  shape_t kernel_shape;
  shape_t strides;
  shape_t dilation;
  shape_t padding;

  CudnnConv2D(shape_t strides, shape_t dilation, shape_t kernel_shape,
              shape_t padding)
      : kernel_shape(kernel_shape), strides(strides), dilation(dilation),
        padding(padding) {}
};

class CudnnPooling2D : public ADPrimitive {
  std::string str() { return "CudnnPooling2D<" + reduce_type + ">"; }
  DEFINE_DISPATCH_CUDA
  // CUDNN ATTRIBUTES
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnPoolingDescriptor_t pooling_desc;
  int workspace_size;
  bool initialized = false;
  void *workspace = nullptr;

public:
  shape_t kernel_shape;
  shape_t strides;
  std::string reduce_type;

  CudnnPooling2D(shape_t kernel_shape, shape_t strides, std::string reduce_type)
      : kernel_shape(kernel_shape), strides(strides), reduce_type(reduce_type) {
  }
};

class CudnnPooling2DVjp : public ADPrimitive {
  std::string str() { return "CudnnPooling2DVjp<" + reduce_type + ">"; }
  DEFINE_DISPATCH_CUDA
  // CUDNN ATTRIBUTES
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t forward_input_desc;
  cudnnTensorDescriptor_t forward_output_desc;
  cudnnTensorDescriptor_t out_grad_desc;
  cudnnTensorDescriptor_t in_grad_desc;
  cudnnPoolingDescriptor_t pooling_desc;
  int workspace_size;
  bool initialized = false;
  void *workspace = nullptr;

public:
  shape_t kernel_shape;
  shape_t strides;
  std::string reduce_type;

  CudnnPooling2DVjp(shape_t kernel_shape, shape_t strides,
                    std::string reduce_type)
      : kernel_shape(kernel_shape), strides(strides), reduce_type(reduce_type) {
  }
};
// optimized backward pass for cudnn conv2d for the weight
class CudnnConv2dVjpWeight : public ADPrimitive {
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(CudnnConv2dVjp)
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_grad_desc;
  cudnnFilterDescriptor_t weight_grad_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  size_t workspace_size;
  cudnnConvolutionBwdFilterAlgo_t algo;
  bool initialized = false;
  void *workspace = nullptr;

public:
  shape_t kernel_shape;
  shape_t strides;
  shape_t dilation;
  shape_t padding;

  CudnnConv2dVjpWeight(shape_t strides, shape_t dilation, shape_t kernel_shape,
                       shape_t padding)
      : kernel_shape(kernel_shape), strides(strides), dilation(dilation),
        padding(padding) {}
};

class CudnnConv2dVjpInput : public ADPrimitive {
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(CudnnConv2dVjp)
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_grad_desc;
  cudnnTensorDescriptor_t output_grad_desc;
  cudnnFilterDescriptor_t weight_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  size_t workspace_size;
  cudnnConvolutionBwdDataAlgo_t algo;
  bool initialized = false;
  void *workspace = nullptr;

public:
  shape_t kernel_shape;
  shape_t strides;
  shape_t dilation;
  shape_t padding;

  CudnnConv2dVjpInput(shape_t strides, shape_t dilation, shape_t kernel_shape,
                      shape_t padding)
      : kernel_shape(kernel_shape), strides(strides), dilation(dilation),
        padding(padding) {}
};

class CudnnLRN : public ADPrimitive {
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(CudnnLRN)

  // CUDNN ATTRIBUTES
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnLRNDescriptor_t lrn_desc;
  bool initialized = false;

public:
  int size;
  double alpha;
  double beta;
  double k;

  CudnnLRN(int size, double alpha, double beta, double k)
      : size(size), alpha(alpha), beta(beta), k(k) {}
};

class CudnnLRNVjpInput : public ADPrimitive {
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(CudnnLRNVjpInput)
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_grad_desc;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_grad_desc;
  cudnnTensorDescriptor_t forward_output_desc;
  cudnnTensorDescriptor_t in_grad_desc;
  cudnnLRNDescriptor_t lrn_desc;
  bool initialized = false;

public:
  int size;
  double alpha;
  double beta;
  double k;

  CudnnLRNVjpInput(int size, double alpha, double beta, double k)
      : size(size), alpha(alpha), beta(beta), k(k) {}
};

class CudnnSdpa : public ADPrimitive {
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(CudnnSdpa)

  // CUDNN ATTRIBUTES
  cudnnHandle_t handle;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnActivationDescriptor_t act_desc;
  bool initialized = false;
};

class FromFunctions : public ADPrimitive {

  std::function<std::vector<Tensor>(const std::vector<Tensor> &inputs)>
      _forward_fn;
  std::function<std::vector<Tensor>(const std::vector<Tensor> &primals,
                                    const std::vector<Tensor> &tangents,
                                    const std::vector<Tensor> &outputs)>
      _backward_fn;
  std::function<std::vector<View>(const std::vector<Tensor> &inputs)>
      _precompute;
  std::string name;

public:
  explicit FromFunctions(
      std::function<std::vector<Tensor>(const std::vector<Tensor> &inputs)>
          forward_fn,
      std::function<std::vector<Tensor>(const std::vector<Tensor> &primals,
                                        const std::vector<Tensor> &tangents,
                                        const std::vector<Tensor> &outputs)>
          backward_fn,
      std::optional<
          std::function<std::vector<Tensor>(const std::vector<Tensor> &inputs)>>
          precompute_fn = std::nullopt,
      std::string name = "")
      : _forward_fn(forward_fn), _backward_fn(backward_fn), name(name) {
    if (precompute_fn.has_value()) {
      _precompute =
          [this, precompute_fn](
              const std::vector<Tensor> &inputs) -> std::vector<View> {
        auto ret = precompute_fn.value()(inputs);
        // return the views of the tensors
        std::vector<View> views;
        views.reserve(ret.size());
        for (auto &t : ret) {
          views.push_back(t.view());
        }
        return views;
      };
    } else {
      _precompute = [this](const std::vector<Tensor> &inputs) {
        return default_precompute(inputs);
      };
    }
  }

  void dispatch_general(const std::vector<Tensor> &inputs,
                        std::vector<Tensor> &outputs) {
    std::vector<Tensor> outs = _forward_fn(inputs);

    for (size_t i = 0; i < outs.size(); i++) {
      outs[i].eval(false);
    }

    PG_CHECK_RUNTIME(outs.size() == outputs.size(), "Expected ", outputs.size(),
                     " outputs, got ", outs.size());
    for (size_t i = 0; i < outs.size(); i++) {
      outputs[i].assign(outs[i]);
    }
  }

  void dispatch_cpu(const std::vector<Tensor> &inputs,
                    std::vector<Tensor> &outputs) override {
    dispatch_general(inputs, outputs);
  }

  void dispatch_cuda(const std::vector<Tensor> &inputs,
                     std::vector<Tensor> &outputs) override {
    dispatch_general(inputs, outputs);
  }

  std::vector<Tensor> backward(const std::vector<Tensor> &primals,
                               const std::vector<Tensor> &tangents,
                               const std::vector<Tensor> &outputs) override {
    return _backward_fn(primals, tangents, outputs);
  }

  std::vector<View> default_precompute(const std::vector<Tensor> &inputs) {
    auto outs = _forward_fn(inputs);
    std::vector<View> views;
    views.reserve(outs.size());
    for (auto &out : outs) {
      views.push_back(out.view());
    }

    return views;
  }

  std::vector<View> precompute(const std::vector<Tensor> &inputs) override {
    return _precompute(inputs);
  }

  std::string str() {
    return name == "" ? "FromFunctions" : "Custom<" + name + ">";
  }
};

class UnaryPrimitive : public ADPrimitive {};

class FromNumpy : public ADPrimitive {
protected:
  shape_t _shape;
  std::shared_ptr<device::Device> _device;
  DType _dtype;
  strides_t _strides;
  void *_data_ptr;
  size_t _buffer_size;
  void _dispatch_general(std::vector<Tensor> &outputs,
                         std::shared_ptr<device::Device> device);

public:
  bool eager() { return true; }
  explicit FromNumpy(shape_t shape, DType dtype, strides_t strides,
                     void *data_ptr, size_t buffer_size,
                     std::shared_ptr<device::Device> device)
      : _shape(shape), _dtype(dtype), _strides(strides), _data_ptr(data_ptr),
        _buffer_size(buffer_size), _device(device) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(FromNumpy)
  DEFINE_PRECOMPUTE
};
class Log : public UnaryPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Log)
  DEFINE_PRECOMPUTE
};

class Exp : public UnaryPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Exp)
  DEFINE_PRECOMPUTE
};

class Sin : public UnaryPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Sin)
  DEFINE_PRECOMPUTE
};

class Cos : public UnaryPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Cos)
  DEFINE_PRECOMPUTE
};

class BinaryPrimitive : public ADPrimitive {};

class Add : public BinaryPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Add)
  DEFINE_PRECOMPUTE
};

class Mul : public BinaryPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Mul)
  DEFINE_PRECOMPUTE
};

class Sub : public BinaryPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Sub)
  DEFINE_PRECOMPUTE
};

class Div : public BinaryPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Div)
  DEFINE_PRECOMPUTE
};

class Gt : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Gt)
  DEFINE_PRECOMPUTE
};

class Lt : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Lt)
  DEFINE_PRECOMPUTE
};

class Eq : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Eq)
  DEFINE_PRECOMPUTE
};

class Neq : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Neq)
  DEFINE_PRECOMPUTE
};

class Ge : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Ge)
  DEFINE_PRECOMPUTE
};

class Le : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Le)
  DEFINE_PRECOMPUTE
};

class Pow : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Pow)
  DEFINE_PRECOMPUTE
};

class Max : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Max)
  DEFINE_PRECOMPUTE
};

class Where : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Where)
  DEFINE_PRECOMPUTE
};

// REDUCE
class Reduce : public ADPrimitive {
protected:
  axes_t _axes;
  bool _keepdims;
  int _total_out_numel;
  int _total_reduce_numel;
  shape_t reduced_shape_assuming_keepdims;
  // out * reduce = in numel
  const shape_t
  _reduce_single_shape_assuming_keepdims(View &input_view,
                                         std::vector<axis_t> axes) {
    shape_t shape = shape_t(input_view.shape()); // copy the shape
    for (size_t i = 0; i < axes.size(); i++) {
      int single_axis = axes[i];
      if (single_axis < 0) {
        single_axis = shape.size() + single_axis;
        PG_CHECK_ARG(single_axis < shape.size(), "axis out of bounds, got ",
                     single_axis, " for shape ", vec_to_string(shape));
      }
      shape[single_axis] = 1;
    }

    return shape;
  }

  const shape_t _reduce_single_shape_assuming_keepdims(View &input_view,
                                                       axis_t axis) {

    shape_t shape = shape_t(input_view.shape()); // copy the shape
    if (axis < 0) {
      axis = shape.size() + axis;
    }
    PG_CHECK_ARG(axis < shape.size(), "axis out of bounds, got ", axis,
                 " for shape ", vec_to_string(shape));
    shape[axis] = 1;

    return shape;
  }

public:
  DEFINE_STR_NAME(Reduce)
  explicit Reduce(axes_t axes, bool keepdims)
      : _axes(axes), _keepdims(keepdims) {}
  axes_t axes() { return _axes; }
  int total_out_numel() { return _total_out_numel; }
  int total_reduce_numel() { return _total_reduce_numel; }
};

class Sum : public Reduce {
  using Reduce::Reduce;

public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_BACKWARD
  DEFINE_STR_NAME(Sum)
  DEFINE_PRECOMPUTE
};

class MaxReduce : public Reduce {
  using Reduce::Reduce;

public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(MaxReduce)
  DEFINE_BACKWARD
  DEFINE_PRECOMPUTE
};

class Mean : public Reduce {
  using Reduce::Reduce;

public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Mean)
  DEFINE_BACKWARD

  DEFINE_PRECOMPUTE
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
  DEFINE_PRECOMPUTE
  shape_t shape_to() const { return _shape_to; }
  void set_shape_to(shape_t shape_to) {
    PG_CHECK_RUNTIME(
        _shape_to.size() == shape_to.size(),
        "Tried to modify shape_to, but the new shape has different ndim");
    _shape_to = shape_to;
  }
  const axes_t &get_broadcasted_axes() const { return _broadcasted_axes; }
  const axes_t &get_created_axes() const { return _created_axes; }
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
  DEFINE_PRECOMPUTE
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

  DEFINE_PRECOMPUTE
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

  DEFINE_PRECOMPUTE
};

class MatMul : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(MatMul)
  DEFINE_BACKWARD

  DEFINE_PRECOMPUTE
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

  DEFINE_BACKWARD
  DEFINE_PRECOMPUTE

  shape_t kernel_shape() { return _kernel_shape; }
  shape_t strides() { return _strides; }
  shape_t padding() { return _padding; }
  shape_t dilation() { return _dilation; }
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

  DEFINE_BACKWARD
  DEFINE_PRECOMPUTE

  shape_t output_shape() { return _output_shape; }
  shape_t kernel_shape() { return _kernel_shape; }
  shape_t strides() { return _strides; }
  shape_t padding() { return _padding; }
  shape_t dilation() { return _dilation; }
};

class BilinearResize : public ADPrimitive {
protected:
  shape_t _output_shape;

public:
  explicit BilinearResize(shape_t output_shape) : _output_shape(output_shape) {}
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(BilinearResize)

  DEFINE_PRECOMPUTE

  shape_t output_shape() { return _output_shape; }
};

class OneHotVector : public ADPrimitive {
protected:
  int num_classes;

public:
  explicit OneHotVector(int num_classes) : num_classes(num_classes) {}
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(OneHotVector)
  DEFINE_PRECOMPUTE
  DEFINE_BACKWARD // incorrect, just to avoid crash
};

class Reshape : public ADPrimitive {
protected:
  axes_t _shape_to;

public:
  explicit Reshape(axes_t shape_to) : _shape_to(shape_to) {}
  DEFINE_DISPATCH_CUDA
  DEFINE_DISPATCH_CPU
  DEFINE_STR_NAME(Reshape)

  DEFINE_BACKWARD
  DEFINE_PRECOMPUTE
};

class FusedLinearBiasAct : public ADPrimitive {
public:
  std::string activation;

  explicit FusedLinearBiasAct(std::string activation)
      : activation(activation) {}
  DEFINE_DISPATCH_CUDA
  std::string str() { return "FusedLinearBiasAct<" + activation + ">"; }
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

  DEFINE_STR_NAME(Select)
  DEFINE_BACKWARD
  DEFINE_PRECOMPUTE
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

  DEFINE_BACKWARD
  DEFINE_DISPATCH_CUDA
  DEFINE_PRECOMPUTE
};

class AsContiguous : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(AsContiguous)

  DEFINE_BACKWARD
  DEFINE_PRECOMPUTE
};

class AsType : public ADPrimitive {
protected:
  DType _dtype_to;

public:
  explicit AsType(DType dtype_to) : _dtype_to(dtype_to) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(AsType)

  DEFINE_BACKWARD

  DEFINE_PRECOMPUTE

  DType dtype_to() { return _dtype_to; }
};

class Fill : public ADPrimitive {
protected:
  double _value;
  shape_t _shape;
  DType _dtype;
  std::shared_ptr<device::Device> _device;

public:
  explicit Fill(double value, DType dtype, shape_t shape,
                std::shared_ptr<device::Device> device)
      : _value(value), _shape(shape), _dtype(dtype), _device(device) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  std::string str() { return "Fill<" + std::to_string(_value) + ">"; }

  DEFINE_BACKWARD
  DEFINE_PRECOMPUTE

  double value() { return _value; }
};

class Binomial : public ADPrimitive {
protected:
  double _p;
  shape_t _shape;
  DType _dtype;
  std::shared_ptr<device::Device> _device;

public:
  explicit Binomial(double p, shape_t shape, DType dtype,
                    std::shared_ptr<device::Device> device)
      : _p(p), _shape(shape), _dtype(dtype), _device(device) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Binomial)
  DEFINE_PRECOMPUTE
  std::vector<shape_t> infer_output_shapes(const std::vector<Tensor> &inputs) {
    return {_shape};
  }
  std::vector<DType> infer_output_dtypes(const std::vector<Tensor> &inputs) {
    return {_dtype};
  }
  std::vector<Tensor> backward(const std::vector<Tensor> &primals,
                               const std::vector<Tensor> &tangents,
                               const std::vector<Tensor> &outputs) {
    return {};
  }

  double p() { return _p; }
};

class Copy : public ADPrimitive {
public:
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(Copy)
  DEFINE_PRECOMPUTE
  DEFINE_BACKWARD
  explicit Copy() {}
};

class ToDevice : public ADPrimitive {
protected:
  std::shared_ptr<device::Device> _device_to;

public:
  explicit ToDevice(std::shared_ptr<device::Device> device_to)
      : _device_to(device_to) {}
  DEFINE_DISPATCH_CPU
  DEFINE_DISPATCH_CUDA
  DEFINE_STR_NAME(ToDevice)

  DEFINE_BACKWARD
  DEFINE_PRECOMPUTE
  std::shared_ptr<device::Device> device_to() { return _device_to; }
};

} // namespace pg