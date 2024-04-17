#include "ad_primitives.hpp"
#include "./binary_helpers.hpp"
#include "./matmul_helpers.hpp"
#include "./reduce_helpers.hpp"
#include "./unary_helpers.hpp"
#include "common/view_helpers.hpp"
#include "cpu/view_helpers.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#define DECL_BINARY_OP(NAME, OP)                                               \
  void NAME::dispatch_cpu(const std::vector<Tensor> &inputs,                   \
                          std::vector<Tensor> &outputs) {                      \
    CHECK_INPUTS_LENGTH(inputs, 2);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                          \
    const Tensor &a = inputs[0];                                               \
    const Tensor &b = inputs[1];                                               \
    CHECK_SAME_SHAPE(a, b);                                                    \
    outputs[0].init_view(                                                      \
        std::make_shared<View>(a.shape(), a.dtype(), device::CPU));            \
    cpu::dispatch_binary_op(a.shape(), a.strides(), b.strides(),               \
                            outputs[0].strides(), a.get_base_ptr(),            \
                            b.get_base_ptr(), outputs[0].get_base_ptr(),       \
                            a.dtype(), cpu::BinaryOpType::OP);                 \
  }
namespace pg {
    DECL_BINARY_OP(Add, Add)
        DECL_BINARY_OP(Mul, Mul)
        DECL_BINARY_OP(Sub, Sub)
        DECL_BINARY_OP(Div, Div)
        DECL_BINARY_OP(Gt, Gt)
        DECL_BINARY_OP(Lt, Lt)
        DECL_BINARY_OP(Eq, Eq)
        DECL_BINARY_OP(Neq, Neq)
        DECL_BINARY_OP(Ge, Ge)
        DECL_BINARY_OP(Le, Le)
        DECL_BINARY_OP(Pow, Pow)
        DECL_BINARY_OP(Max, Max)

        void Log::dispatch_cpu(const std::vector<Tensor>& inputs,
            std::vector<Tensor>& outputs) {
        CHECK_INPUTS_LENGTH(inputs, 1);
        CHECK_OUTPUTS_LENGTH(outputs, 1);
        const Tensor& a = inputs[0];
        outputs[0].init_view(
            std::make_shared<View>(a.shape(), a.dtype(), device::CPU));
        cpu::dispatch_unary_op(
            a.dtype(), cpu::UnaryOpType::Log, a.get_base_ptr(),
            outputs[0].get_base_ptr(),
            a.nbytes() / dtype_to_size(a.dtype())); // todo -- this assumes contiguity
    }

#define DECL_REDUCE_OP(NAME, OP)                                               \
  void NAME::dispatch_cpu(const std::vector<Tensor> &inputs,                   \
                          std::vector<Tensor> &outputs) {                      \
    CHECK_INPUTS_LENGTH(inputs, 1);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                          \
    const Tensor &a = inputs[0];                                               \
    const axes_t &axes = _axes;                                                \
    if (axes.size() == 0) {                                                    \
      throw std::runtime_error("Reduce expects at least one axis");            \
    }                                                                          \
    const bool keepdims = _keepdims;                                           \
    const bool is_sum = OP == cpu::ReduceOp::Sum;                              \
    View old_view = inputs[0].view();                                          \
    View new_view;                                                             \
    for (int i = 0; i < axes.size(); i++) {                                    \
      const shape_t new_shape =                                                \
          _reduce_single_shape_assuming_keepdims(old_view, axes[i]);           \
      new_view = View(new_shape, a.dtype(), device::CPU);                      \
      axis_t axis = axes[i];                                                   \
      cpu::dispatch_reduce(old_view.get_base_ptr(), new_view.get_base_ptr(),   \
                           old_view.strides(), old_view.shape(), axis,         \
                           old_view.dtype(), OP);                              \
      old_view = new_view;                                                     \
    }                                                                          \
    if (!keepdims) { /* squeeze the axes if keepdims is false*/                \
      new_view = view::squeeze(old_view, axes);                                \
    }                                                                          \
    outputs[0].init_view(std::make_shared<View>(new_view));                    \
  }

    DECL_REDUCE_OP(Sum, cpu::ReduceOp::Sum)
        DECL_REDUCE_OP(MaxReduce, cpu::ReduceOp::Max)
        DECL_REDUCE_OP(Mean, cpu::ReduceOp::Mean)

        void MatMul::dispatch_cpu(const std::vector<Tensor>& inputs,
            std::vector<Tensor>& outputs) {
        CHECK_INPUTS_LENGTH(inputs, 2);
        CHECK_OUTPUTS_LENGTH(outputs, 1);
        View a = inputs[0].view();
        View b = inputs[1].view();
        PG_CHECK_ARG(a.dtype() == b.dtype(),
            "MatMul expects inputs to have the same dtype, got ",
            dtype_to_string(a.dtype()), " and ", dtype_to_string(b.dtype()));

        bool added_a_dim = false;
        bool added_b_dim = false;
        DType dtype = a.dtype();
        // In vector-matrix multiplication, expand dimensions to make both inputs
        // matrices
        if (a.ndim() == 1 && b.ndim() != 1) {
            added_a_dim = true;
            a = view::unsqueeze(a, 0);
        }
        else {
            a = cpu::view::as_contiguous(a);
        }
        // In matrix-vector multiplication, expand dimensions to make both inputs
        // matrices
        if (b.ndim() == 1 && a.ndim() != 1) {
            added_b_dim = true;
            b = view::unsqueeze(b, 1);
        }
        else {
            b = cpu::view::as_contiguous(b);
        }
        shape_t new_shape;
        size_t size1, midsize, size2;
        if (a.ndim() == 1 && b.ndim() == 1) {
            View out_view({ 1 }, a.dtype(), device::CPU);
            dispatch_contiguous_dot_ker(a.get_base_ptr(), b.get_base_ptr(),
                out_view.get_base_ptr(), a.shape()[0], dtype);
            out_view = view::squeeze(out_view); // Make a scalar output
            outputs[0].init_view(std::make_shared<View>(out_view));
            return;
        }
        else {
            if (a.ndim() > 2 || b.ndim() > 2) {
            size_t a_prod = 1;
            size_t b_prod = 1;
            for (size_t i = 0; i < a.ndim(); i++) {
                a_prod *= a.shape()[i];
            }
            for (size_t i = 0; i < b.ndim(); i++) {
                b_prod *= b.shape()[i];
            }
            if (a.ndim() > b.ndim()) { // we will try to broadcast, but keep
                // last to dims. That is, broadcast D0...DN where
                // shapes are D0...DN, M, N
                shape_t b_new = shape_t(a.shape());
                b_new[b_new.size() - 1] = b.shape()[b.shape().size() - 1];
                b_new[b_new.size() - 2] = b.shape()[b.shape().size() - 2];
                b = cpu::view::as_contiguous(std::get<0>(view::broadcasted_to(b, b_new)));
            }
            else if (a.ndim() < b.ndim()) {
                shape_t a_new = shape_t(b.shape());
                a_new[a_new.size() - 1] = a.shape()[a.shape().size() - 1];
                a_new[a_new.size() - 2] = a.shape()[a.shape().size() - 2];
                a = cpu::view::as_contiguous(std::get<0>(view::broadcasted_to(a, a_new)));
                // if ndim are equal, we will broadcast the one with the smallest product
            }
            else if (a_prod >= b_prod) {
                shape_t b_new = shape_t(a.shape());
                b_new[b_new.size() - 1] = b.shape()[b.shape().size() - 1];
                b_new[b_new.size() - 2] = b.shape()[b.shape().size() - 2];
                b = cpu::view::as_contiguous(std::get<0>(view::broadcasted_to(b, b_new)));
            }
            else if (a_prod < b_prod) {
                shape_t a_new = shape_t(b.shape());
                a_new[a_new.size() - 1] = a.shape()[a.shape().size() - 1];
                a_new[a_new.size() - 2] = a.shape()[a.shape().size() - 2];
                a = cpu::view::as_contiguous(std::get<0>(view::broadcasted_to(a, a_new)));
            }
            }
            size1 = a.shape().at(a.ndim() - 2);
            midsize = a.shape().at(a.ndim() - 1);
            size2 = b.shape().at(b.ndim() - 1);
            new_shape = a.shape();
            new_shape[new_shape.size() - 1] = size2;
            if (added_a_dim) {
                new_shape.erase(new_shape.begin());
            }
            if (added_b_dim) {
                new_shape.erase(new_shape.end() - 1);
            }
            int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                std::multiplies<int>());

            View out_view(new_shape, dtype, device::CPU);
            size_t M = size1;
            size_t N = size2;
            size_t K = midsize;
            size_t B = new_size / (M * N); // batch size
            std::cout << "M: " << M << " N: " << N << " K: " << K << " B: " << B << std::endl;
            dispatch_contiguous_matmul_ker(a.get_base_ptr(), b.get_base_ptr(),
                out_view.get_base_ptr(), M, N, K, B, dtype);
            outputs[0].init_view(std::make_shared<View>(out_view));
        }
    }
}