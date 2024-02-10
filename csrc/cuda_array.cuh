#pragma once

#include <iostream>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
typedef void (*BinaryOpKernel)(const int *strides, const int *ostrides,
                               const int *shape, const int ndim, const float *a,
                               const float *b, float *out);

typedef void (*ElementWiseOpKernel)(const int *in_strides, const int *shape,
                                    const int num_dims, const float *in,
                                    float *out);

typedef void (*TernaryOpKernel)(
    const int *first_strides,  /* in bytes */
    const int *second_strides, /* in bytes */
    const int *third_strides,  /* in bytes */
    const int *shape,   /* both lhs and rhs should have equal shape, we dont \
                           handle broadcasting here */
    const int num_dims, /* equals len of strides and shape */
    const float *first, const float *second, const float *third, float *out);

using ShapeLike = std::vector<size_t>;

class CudaArray {
public:
  bool isContiguous() const;
  std::shared_ptr<float> ptr;
  size_t size;
  ShapeLike shape;
  ShapeLike strides;

  CudaArray(size_t size, const ShapeLike &shape, const ShapeLike &strides,
            const std::shared_ptr<float> &sharedPtr);
  CudaArray(size_t size, ShapeLike shape, ShapeLike strides);
  CudaArray(size_t size, ShapeLike shape);

  ~CudaArray();
  CudaArray(const CudaArray &other);
  CudaArray &operator=(const CudaArray &other);
  CudaArray(CudaArray &&other);
  CudaArray &operator=(CudaArray &&other);
  CudaArray clone() const;

  CudaArray broadcastTo(const ShapeLike _shape) const;
  CudaArray binop(const CudaArray &other, BinaryOpKernel Ker) const;
  CudaArray ternaryop(const CudaArray &second, const CudaArray &third,
                      TernaryOpKernel Ker) const;
  CudaArray elwiseop(ElementWiseOpKernel Ker) const;
  float getitem(ShapeLike index) const;
  int ndim() const;
  CudaArray matMul(const CudaArray &other) const;
  CudaArray permute(ShapeLike axes) const;
  CudaArray asContiguous() const;

  static CudaArray fromNumpy(py::array_t<float> np_array);
  py::array_t<float> toNumpy() const;

  std::string toString() const;
};