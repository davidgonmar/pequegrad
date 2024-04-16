#pragma once
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

} // namespace pg