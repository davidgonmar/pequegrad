#include "reduce.hpp"
#include "ad_primitives.hpp"
#include "dispatch.hpp"

#define FUNCTOR_FOR_OP_KIND(OP)                                                \
  using functor_t = typename std::conditional<                                 \
      OP == cpu::ReduceOp::Sum, cpu::SumOp<scalar_t>,                          \
      typename std::conditional<OP == cpu::ReduceOp::Max,                      \
                                cpu::MaxOp<scalar_t>,                          \
                                cpu::MeanOp<scalar_t>>::type>::type

#define DEF_REDUCE_OP(NAME, OP)                                                \
  void NAME::dispatch_cpu(const std::vector<Tensor> &inputs,                   \
                          std::vector<Tensor> &outputs) {                      \
    CHECK_INPUTS_LENGTH(inputs, 1);                                            \
    CHECK_OUTPUTS_LENGTH(outputs, 1);                                          \
    const Tensor &a = inputs[0];                                               \
    const axes_t &axes = _axes;                                                \
    const bool keepdims = _keepdims;                                           \
    const bool is_sum = OP == cpu::ReduceOp::Sum;                              \
    View old_view = inputs[0].view();                                          \
    View new_view;                                                             \
    for (int i = 0; i < axes.size(); i++) {                                    \
      const shape_t new_shape =                                                \
          _reduce_single_shape_assuming_keepdims(old_view, axes[i]);           \
      new_view = View(new_shape, a.dtype(), device::CPU);                      \
      axis_t axis = axes[i] < 0 ? old_view.ndim() + axes[i] : axes[i];         \
      PG_DISPATCH_ALL_TYPES(a.dtype(), "reduce_cpu", [&] {                     \
        FUNCTOR_FOR_OP_KIND(OP);                                               \
        cpu::reduce_base_fn<functor_t, scalar_t>(                              \
            old_view.get_casted_base_ptr<scalar_t>(),                          \
            new_view.get_casted_base_ptr<scalar_t>(), old_view.strides(),      \
            old_view.shape(), axis);                                           \
      });                                                                      \
      old_view = new_view;                                                     \
    }                                                                          \
    if (!keepdims) { /* squeeze the axes if keepdims is false*/                \
      new_view = view::squeeze(old_view, axes);                                \
    }                                                                          \
    outputs[0].init_view(std::make_shared<View>(new_view));                    \
  }

namespace pg {

DEF_REDUCE_OP(Sum, cpu::ReduceOp::Sum)
DEF_REDUCE_OP(MaxReduce, cpu::ReduceOp::Max)
DEF_REDUCE_OP(Mean, cpu::ReduceOp::Mean)

} // namespace pg