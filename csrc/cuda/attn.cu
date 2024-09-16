#include <cudnn.h>
#include <cudnn_frontend.h>

#include "ad_primitives.hpp"
#include "cuda_utils.cuh"
#include "dtype.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <iostream>

namespace pg {

namespace fe = cudnn_frontend;
using graph_and_tensors =
    std::tuple<std::shared_ptr<fe::graph::Graph>,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Q,
               std::shared_ptr<fe::graph::Tensor_attributes>, // K,
               std::shared_ptr<fe::graph::Tensor_attributes>, // V,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Attn_scale,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Seed,
               std::shared_ptr<fe::graph::Tensor_attributes>, // Offset,
               std::shared_ptr<fe::graph::Tensor_attributes>, // O
               std::shared_ptr<fe::graph::Tensor_attributes>  // Stats
               >;
#define MHA_DIM 4

struct MHAParams {
  fe::DataType_t dataType;
  std::array<int, MHA_DIM> q_dim;
  std::array<int, MHA_DIM> k_dim;
  std::array<int, MHA_DIM> v_dim;
  std::array<int, MHA_DIM> q_stride;
  std::array<int, MHA_DIM> k_stride;
  std::array<int, MHA_DIM> v_stride;
  int64_t b;
  int64_t h;
  int64_t s_q;
  int64_t s_kv;
  int64_t d_qk;
  int64_t d_v;
  double dropout_probability;
  bool is_causal;
  bool return_softmaxstats;
};

template <typename T>
std::vector<int64_t> transformToInt64Vector(const std::vector<T> &input) {
  std::vector<int64_t> output;
  output.reserve(input.size());
  for (const auto &elem : input) {
    output.push_back(static_cast<int64_t>(elem));
  }
  return output;
}

template <typename T>
std::vector<int32_t> transformToInt32Vector(const std::vector<T> &input) {
  std::vector<int32_t> output;
  output.reserve(input.size());
  for (const auto &elem : input) {
    output.push_back(static_cast<int32_t>(elem));
  }
  return output;
}

// taken from pytorch
auto fixSizeOneDimStrideSDPA(const shape_t &sizes,
                             std::vector<int64_t> _strides) {
  std::vector<int64_t> strides = std::vector<int64_t>(_strides.size(), 0);
  int dims = sizes.size();
  for (int d = 0; d < dims; d++) {
    int64_t curr_stride = _strides[d] / sizeof(float); // assuming float
    if (sizes[d] == 1 && !curr_stride) {
      curr_stride = 1;
      for (int d2 = d + 1; d2 < dims; d2++) {
        curr_stride *= _strides[d2] / sizeof(float); // assuming float
      }
      strides[d] = curr_stride;
    } else {
      strides[d] = curr_stride; // assuming float
    }
  }
  // print
  for (int i = 0; i < strides.size(); i++) {
    std::cout << "Stride " << i << " : " << strides[i] << std::endl;
  }
  return strides;
}

void setMHAParams(MHAParams &params, int64_t b, int64_t h, int64_t s_q,
                  int64_t s_kv, int64_t d_qk, int64_t d_v, const Tensor &q,
                  const Tensor &k, const Tensor &v, double dropout_probability,
                  bool is_causal, bool return_softmaxstats) {
  std::cout << "Setting MHA Para 22ms" << std::endl;
  memset(&params, 0, sizeof(MHAParams));
  params.dataType = fe::DataType_t::HALF;
  params.b = b;
  params.h = h;
  params.d_qk = d_qk;
  params.d_v = d_v;
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.dropout_probability = dropout_probability;
  params.is_causal = is_causal;
  params.return_softmaxstats = return_softmaxstats;
  PG_CHECK_RUNTIME(q.ndim() == MHA_DIM, "Q tensor should have 4 dims, got ",
                   q.ndim());
  PG_CHECK_RUNTIME(k.ndim() == MHA_DIM, "K tensor should have 4 dims, got ",
                   k.ndim());
  PG_CHECK_RUNTIME(v.ndim() == MHA_DIM, "V tensor should have 4 dims, got ",
                   v.ndim());
  std::cout << "Setting MHA Params" << std::endl;
  auto qshape = transformToInt32Vector(q.shape());
  auto kshape = transformToInt32Vector(k.shape());
  auto vshape = transformToInt32Vector(v.shape());
  auto qstrides = transformToInt32Vector(q.strides());
  auto kstrides = transformToInt32Vector(k.strides());
  auto vstrides = transformToInt32Vector(v.strides());
  std::copy(qshape.begin(), qshape.end(), params.q_dim.data());
  std::copy(qstrides.begin(), qstrides.end(), params.q_stride.data());
  std::copy(kshape.begin(), kshape.end(), params.k_dim.data());
  std::copy(kstrides.begin(), kstrides.end(), params.k_stride.data());
  std::copy(vshape.begin(), vshape.end(), params.v_dim.data());
  std::copy(vstrides.begin(), vstrides.end(), params.v_stride.data());
  std::cout << "MHA Params set" << std::endl;
}

#define PG_CHECK_CUDNN_FE(expr)                                                \
  if (!expr.is_good()) {                                                       \
    throw std::runtime_error("CUDNN_FE Error: " + std::string(expr.err_msg));  \
  }

auto build_graph_and_tensors(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                             int64_t d_qk, int64_t d_v, float scaling_factor,
                             bool is_causal, double dropout_probability,
                             const Tensor &q, const Tensor &k, const Tensor &v,
                             Tensor &o, int32_t dpseed, int32_t dpoffset,
                             cudnnHandle_t &handle) {
  auto dtype = fe::DataType_t::FLOAT;
  auto mha_graph = std::make_shared<fe::graph::Graph>();
  mha_graph->set_io_data_type(dtype)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  auto Q = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("q")
          .set_dim(transformToInt64Vector(q.shape()))
          .set_stride(fixSizeOneDimStrideSDPA(
              q.shape(), transformToInt64Vector(q.strides()))));
  auto K = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("k")
          .set_dim(transformToInt64Vector(k.shape()))
          .set_stride(fixSizeOneDimStrideSDPA(
              k.shape(), transformToInt64Vector(k.strides()))));
  auto V = mha_graph->tensor(
      fe::graph::Tensor_attributes()
          .set_name("v")
          .set_dim(transformToInt64Vector(v.shape()))
          .set_stride(fixSizeOneDimStrideSDPA(
              v.shape(), transformToInt64Vector(v.strides()))));
  auto attn_scale =
      mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

  auto seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("seed")
                                    .set_dim({1, 1, 1, 1})
                                    .set_stride({1, 1, 1, 1})
                                    .set_is_pass_by_value(true)
                                    .set_data_type(fe::DataType_t::INT32));
  auto offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("offset")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_is_pass_by_value(true)
                                      .set_data_type(fe::DataType_t::INT32));
  auto scaled_dot_product_flash_attention_options =
      fe::graph::SDPA_attributes()
          .set_name("CUDNN_SDPA")
          .set_is_inference(true) // return softmax is false
          .set_causal_mask(is_causal)
          .set_attn_scale(attn_scale)
          .set_dropout(dropout_probability, seed, offset);

  auto seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("Seq_q")
                                     .set_dim({b, 1, 1, 1})
                                     .set_stride({1, 1, 1, 1})
                                     .set_data_type(fe::DataType_t::INT32));
  auto seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("Seq_kv")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));

  auto [O, Stats] =
      mha_graph->sdpa(Q, K, V, scaled_dot_product_flash_attention_options);
  O->set_output(true)
      .set_dim(transformToInt64Vector(o.shape()))
      .set_stride(fixSizeOneDimStrideSDPA(o.shape(),
                                          transformToInt64Vector(o.strides())));

  if (Stats) {
    Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);
  }

  PG_CHECK_CUDNN_FE(mha_graph->validate());
  PG_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
  PG_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
  PG_CHECK_CUDNN_FE(mha_graph->check_support(handle));
  PG_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

  return std::make_tuple(std::move(mha_graph), std::move(Q), std::move(K),
                         std::move(V), std::move(attn_scale), std::move(seed),
                         std::move(offset), std::move(O), std::move(Stats));
}

void run_cudnn_SDP_fprop(int64_t b, int64_t h, int64_t s_q, int64_t s_kv,
                         int64_t d_qk, int64_t d_v, float scaling_factor,
                         bool is_causal, double dropout_probability,
                         const Tensor &q, const Tensor &k, const Tensor &v,
                         Tensor &o, int32_t dropoutseed,
                         int32_t dropoutoffset) {
  cudnnHandle_t handle;
  cudnnCreate(&handle);
  std::cout << "Handle created" << std::endl;
  graph_and_tensors graph_and_tensors_values = build_graph_and_tensors(
      b, h, s_q, s_kv, d_qk, d_v, scaling_factor, is_causal,
      dropout_probability, q, k, v, o, dropoutseed, dropoutoffset, handle);
  std::cout << "Graph and tensors built" << std::endl;
  auto [mha_graph, Q, K, V, attn_scale, seed, offset, O, Stats] =
      graph_and_tensors_values;
  std::cout << "Graph and tensors unpacked" << std::endl;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
      variant_pack = {{Q, q.get_base_ptr()}, {K, k.get_base_ptr()},
                      {V, v.get_base_ptr()}, {attn_scale, &scaling_factor},
                      {seed, &dropoutseed},  {offset, &dropoutoffset},
                      {O, o.get_base_ptr()}};
  auto workspace_size = mha_graph->get_workspace_size();
  void *workspace_ptr;
  cudaMalloc(&workspace_ptr, workspace_size);
  std::cout << "Workspace allocated" << std::endl;
  auto res = mha_graph->execute(handle, variant_pack, workspace_ptr);
  if (!res.is_good()) {
    std::cout << "Error in executing graph: " << res.err_msg << std::endl;
  }
}

void CudnnSdpa::dispatch_cuda(const std::vector<Tensor> &inputs,
                              std::vector<Tensor> &outputs) {
  PG_CHECK_RUNTIME(inputs.size() == 3, "Expected 3 inputs, got ",
                   inputs.size());
  PG_CHECK_RUNTIME(outputs.size() == 1, "Expected 1 output, got ",
                   outputs.size());
  std::cout << "Dispatching to CudnnSdpa" << std::endl;
  auto &q = inputs[0];
  auto &k = inputs[1];
  auto &v = inputs[2];
  auto &o = outputs[0];
  std::cout << "Checking tensor dims" << std::endl;
  PG_CHECK_RUNTIME(q.shape().size() == 4, "Q tensor should have 4 dims, got ",
                   q.shape().size());
  PG_CHECK_RUNTIME(k.shape().size() == 4, "K tensor should have 4 dims, got ",
                   k.shape().size());
  PG_CHECK_RUNTIME(v.shape().size() == 4, "V tensor should have 4 dims, got ",
                   v.shape().size());
  std::cout << "Allocating view ptr" << std::endl;
  // default params
  o.view_ptr()->allocate();
  MHAParams params;
  std::cout << "Setting MHA Params" << std::endl;
  setMHAParams(params, q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3],
               q.shape()[3], v.shape()[3], q, k, v, 0.0, false, false);
  std::cout << "Running Cudnn SDP Fprop" << std::endl;
  run_cudnn_SDP_fprop(
      params.b, params.h, params.s_q, params.s_kv, params.d_qk, params.d_v, 1.0,

      params.is_causal, params.dropout_probability, q, k, v, o, 0, 0);
}
} // namespace pg
