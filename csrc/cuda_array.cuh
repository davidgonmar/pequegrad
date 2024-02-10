#pragma once

#include <iostream>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
typedef void (*binary_op_kernel)(const int *strides, const int *ostrides,
                               const int *shape, const int ndim, const float *a,
                               const float *b, float *out);

typedef void (*element_wise_op_kernel)(const int *in_strides, const int *shape,
                                    const int num_dims, const float *in,
                                    float *out);

typedef void (*ternary_op_kernel)(
    const int *first_strides,
    const int *second_strides, 
    const int *third_strides,  
    const int *shape,
    const int num_dims,
    const float *first, const float *second, const float *third, float *out);

using shape_t = std::vector<size_t>;

class CudaArray {
public:
  bool is_contiguous() const;
  std::shared_ptr<float> ptr;
  size_t size;
  shape_t shape;
  shape_t strides;

  CudaArray(size_t size, const shape_t &shape, const shape_t &strides,
            const std::shared_ptr<float> &shared_ptr);
  CudaArray(size_t size, shape_t shape, shape_t strides);
  CudaArray(size_t size, shape_t shape);

  ~CudaArray();
  CudaArray(const CudaArray &other);
  CudaArray &operator=(const CudaArray &other);
  CudaArray(CudaArray &&other);
  CudaArray &operator=(CudaArray &&other);
  CudaArray clone() const;

  CudaArray broadcast_to(const shape_t _shape) const;
  CudaArray binop(const CudaArray &other, binary_op_kernel Ker) const;
  CudaArray ternaryop(const CudaArray &second, const CudaArray &third,
                      ternary_op_kernel ker) const;
  CudaArray elwiseop(element_wise_op_kernel ker) const;
  float getitem(shape_t index) const;
  int ndim() const;
  CudaArray mat_mul(const CudaArray &other) const;
  CudaArray permute(shape_t axes) const;
  CudaArray as_contiguous() const;

  static CudaArray from_numpy(py::array_t<float> np_array);
  py::array_t<float> to_numpy() const;

  std::string to_string() const;
};