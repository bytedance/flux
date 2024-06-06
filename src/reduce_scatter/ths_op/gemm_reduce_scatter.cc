//===- gemm_reduce_scatter.cc ------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "c10/cuda/CUDAGuard.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/runtime_config.h"
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/op_registry.h"
#include "flux/ths_op/util.h"
#include "flux/args/reduce_scatter.h"
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/logging_is_not_google_glog.h>
#include <cuda_runtime_api.h>
#include <ATen/core/jit_type.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <nvshmemx.h>
#include <torch/python.h>
#include "flux/utils.h"

#include "reduce_scatter/ths_op/helper_ops.h"
#include "reduce_scatter/reduce_scatter_barrier_struct.hpp"
#include "flux/ths_op/topo_utils.h"
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
#include "nccl.h"
#endif

namespace bytedance {
namespace flux {
namespace ths_op {
using torch::Tensor;

class GemmRS : public torch::CustomClassHolder {
 private:
  const c10d::ProcessGroup tp_group;
  const int32_t nnodes;
  const int32_t max_m;
  const int32_t n_dim;
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;
  const bool fuse_reduction;

 private:
  const int32_t rank;
  const int32_t world_size;
  const int32_t local_world_size;
  const int32_t local_rank;
  const int32_t node_idx;

 private:
  torch::Tensor output_buffer;
  torch::Tensor reduce_buffer;
  torch::Tensor barrier_buffer;
  torch::Tensor gemm_buffer;
  std::vector<void *> output_scatter_ptrs;
  std::vector<void *> barrier_ptrs;
  bool no_nvlink;
  int sub_world_size;
  c10::cuda::CUDAStream rs_stream_;
  cudaEvent_t event_;
  bool use_1d_ring;
  bool use_p2p_read;
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
  ncclComm_t nccl_comm;
#endif
  void
  init_output_buffer() {
    // update max_m and allocate buffer
    if (get_arch() == _Sm80{} && nnodes > 1) {
      int reduce_m_dim = max_m;
      this->reduce_buffer = nvshmem_create_tensor({reduce_m_dim, n_dim}, output_dtype);
    }
    if (get_arch() == _Sm80{} && nnodes > 1 && from_torch_dtype(this->input_dtype) == _BF16{}) {
      // SM80 does not support the fuse reduction for the bfloat16 data type
      // we have to use the float32 global_red instruction when SM80 && nnodes>1 && input_type=bf16
      // Therefore, in this case, here double the size of the output_buffer.
      this->output_buffer = nvshmem_create_tensor({max_m * 2, n_dim}, output_dtype);
    } else {
      this->output_buffer = nvshmem_create_tensor({max_m, n_dim}, output_dtype);
    }
    for (int i = 0; i < world_size; ++i) {
      output_scatter_ptrs[i] = nvshmem_ptr(output_buffer.data_ptr(), i);
      if (i / nnodes == rank / nnodes) {
        // only check for ranks on the same node
        TORCH_CHECK(
            output_scatter_ptrs[i] != nullptr, "nullptr buffr of rank " + std::to_string(i));
      }
    }
  }

  void
  lazy_init_barrier_buffer(int64_t buffer_size) {
    if ((buffer_size == 0) ||
        (barrier_buffer.defined() && buffer_size <= barrier_buffer.numel())) {
      return;
    }

    barrier_buffer = nvshmem_create_tensor({buffer_size}, c10::ScalarType::Byte);
    for (int i = 0; i < world_size; ++i) {
      barrier_ptrs[i] = nvshmem_ptr(barrier_buffer.data_ptr(), i);
    }
  }

  bool
  has_nvlink() {
    return true;
  }

  bool
  use_1d_ring_or_not() {
    ensure_nvml_init();
    int devid = at::cuda::current_device();
    std::string devname(get_gpu_device_name(devid));
    if (devname != "NVIDIA L20" && world_size == 8) {
      return false;
    }
    return true;
  }

  bool
  use_p2p_read_or_not() {
    ensure_nvml_init();
    int devid = at::cuda::current_device();
    std::string devname(get_gpu_device_name(devid));
    if (devname != "NVIDIA L20") {
      return true;
    }
    return false;
  }

  void
  lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0) {
      return;
    }
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      auto options = input.options().dtype(c10::ScalarType::Byte);
      this->gemm_buffer = torch::empty({buffer_size}, options);
    }
  }

  c10::cuda::CUDAStream
  CreateReduceScatterStream() {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t rs_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&rs_stream, cudaStreamNonBlocking));
    return at::cuda::getStreamFromExternal(rs_stream, at::cuda::current_device());
  }

 public:
  GemmRS(
      c10d::ProcessGroup tp_group,
      int32_t nnodes,
      int32_t max_m,
      int32_t n_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight,
      bool fuse_reduction)
      : tp_group(tp_group),
        nnodes(nnodes),
        max_m(max_m),
        n_dim(n_dim),
        input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight),
        fuse_reduction(fuse_reduction),
        rank(tp_group.getRank()),
        world_size(tp_group.getSize()),
        local_world_size(world_size / nnodes),
        local_rank(rank % local_world_size),
        node_idx(rank / local_world_size),
        output_scatter_ptrs(world_size, nullptr),
        barrier_ptrs(world_size, nullptr),
        no_nvlink(!has_nvlink()),
        rs_stream_(CreateReduceScatterStream()),  // private stream. never dup with gemm stream
        use_1d_ring(use_1d_ring_or_not()),
        use_p2p_read(use_p2p_read_or_not()) {
    TORCH_CHECK(
        rank >= 0 && rank < world_size,
        "invalid rank: " + std::to_string(rank) +
            " and world_size: " + std::to_string(world_size));
    TORCH_CHECK(
        world_size % nnodes == 0,
        "invalid nnodes: world_size[" + std::to_string(world_size) + "] % nnodes[" +
            std::to_string(nnodes) + "] != 0");

    this->init_output_buffer();
    CUDA_CHECK(cudaEventCreate(&event_));
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (nnodes > 1 && no_nvlink) {
      nccl_comm = topo_utils::create_nccl_comm_with_processgroup(tp_group);
    } else {
      nccl_comm = nullptr;
    }
#endif
  }

  ~GemmRS() {
    CUDA_CHECK(cudaEventDestroy(event_));
    CUDA_CHECK(cudaStreamDestroy(rs_stream_));
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (nccl_comm) {
      NCCL_CHECK(ncclCommDestroy(nccl_comm));
    }
#endif
  }

  auto
  get_gemm_meta(bool has_bias) {
    ArchEnum arch = get_arch();
    auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    auto dt_conf = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype);
    auto meta = make_gemm_meta(
        dt_conf,
        arch,
        _ReduceScatter{},
        gemm_layout,
        ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}(),
        None{},
        make_reduce_scatter_meta(
            this->fuse_reduction, nnodes > 1 ? _AcrossNode{}() : _IntraNode{}()));
    return meta;
  }

  RuntimeConfig
  get_rt_conf(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    TORCH_CHECK(input.dim() == 2, "input dim is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0);

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      TORCH_CHECK(bias->dim() == 2, "bias dim is not 2");
      TORCH_CHECK(
          m == bias->size(0),
          "bias dim0 != m: " + std::to_string(bias->size(0)) + " vs " + std::to_string(m));
      TORCH_CHECK(
          n == bias->size(1),
          "bias dim1 != n: " + std::to_string(bias->size(1)) + " vs " + std::to_string(n));
    }

    // row major for streamk, todo: make weight layout an option
    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    FLUX_CHECK_LE(m, this->max_m) << "m-dim greater than maximum possible value";
    FLUX_CHECK_EQ(n, this->n_dim) << "n-dim != expected n_dim";
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";
    return make_runtime_config(m, n, k, make_reduce_scatter_runtime_config(world_size, nnodes));
  }

  void
  forward_gemm_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value());
    auto rt_conf = get_rt_conf(input, weight, bias);
    // get cutlass op
    OpRegistry::OpPtr cutlass_op;
    if (hparams.has_value()) {
      cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    // TODO(houqi.1993) using args instead of envs
    static int num_blocks = get_int_from_env("FLUX_RS_BLOCKS", 12);
    static bool use_barrier_queue = get_bool_from_env("FLUX_RS_USE_BARRIER_QUEUE", false);
    static bool use_gemmk = get_bool_from_env("FLUX_RS_USE_GEMMK", no_nvlink);
    static bool use_cudaMemcpyAsync = get_bool_from_env("FLUX_RS_USE_CUDA_MEMCPY_ASYNC", false);
    static int n_split = get_int_from_env("FLUX_RS_N_SPLIT", 1);
    static bool per_tile_flags = get_bool_from_env("FLUX_RS_PER_TILE_FLAGS", no_nvlink);
    const GemmReduceScatterArguments args{
        .m = rt_conf.m(),
        .n = rt_conf.n(),
        .k = rt_conf.k(),
        .rank = static_cast<int>(this->rank),
        .world_size = static_cast<int>(this->world_size),
        .nnodes = static_cast<int>(this->nnodes),
        .alpha = 1.0f,
        .beta = bias.has_value() ? 1.0f : 0.0f,
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = bias.has_value() ? bias->data_ptr() : nullptr,
        .output_scatter_ptrs = this->output_scatter_ptrs.data(),
        .local_reduce_buffer =
            this->reduce_buffer.defined() ? this->reduce_buffer.data_ptr() : nullptr,
        .barrier_ptrs = this->barrier_ptrs.data(),
        .avail_sms = no_nvlink ? 1 : -1,
        .reduce_scatter_args{
            .reduce_scatter_num_blocks = num_blocks,
            .rs_stream = rs_stream_,
            .event = event_,
            .use_barrier_queue = use_barrier_queue,
            .use_gemmk = use_gemmk,
            .per_tile_flags = per_tile_flags,
            .use_cudaMemcpyAsync = use_cudaMemcpyAsync,
            .n_split = n_split,
            .sub_world_size = this->sub_world_size,
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
            .opaque = nccl_comm,
#else
            .opaque = nullptr,
#endif
            .use_1d_ring = use_1d_ring,
            .use_p2p_read = use_p2p_read,
        }};

    // initialize workspace
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int64_t workspace_size = cutlass_op->get_workspace_size(args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;

    // initialize barrier workspace
    int64_t barrier_workspace_size = cutlass_op->get_barrier_workspace_size(args);
    // printf("barrier_workspace_size:%d \n", barrier_workspace_size);
    barrier_workspace_size = barrier_workspace_size / sizeof(int) * sizeof(PerTileFlags);
    this->lazy_init_barrier_buffer(barrier_workspace_size);

    if ((fuse_reduction && !(meta.arch() == _Sm90{})) || this->no_nvlink) {
      // need to zero buffers;
      zero_buffers();
    }
    cutlass_op->run(args, workspace, stream);
  }  // namespace ths_op

  torch::Tensor
  forward_reduce_scatter_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<UnifiedGemmHParams> hparams) {
    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value());
    auto rt_conf = get_rt_conf(input, weight, bias);

    // get cutlass op
    OpRegistry::OpPtr cutlass_op;
    if (hparams.has_value()) {
      cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      cutlass_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    int m = rt_conf.m();
    int n = rt_conf.n();

    auto full_output = this->output_buffer.slice(0, 0, m);
    if (nnodes > 1 && !no_nvlink) {
      // printf("fuse_reduction:%d \n\n", fuse_reduction);
      auto unified_hparams = cutlass_op->get_runtime_gemm_hparams();
      auto tile_shape = unified_hparams.tile_shape();
      auto [tile_M, tile_N, tile_K] = tile_shape;
      int m_rank = m / world_size;
      auto result = torch::empty({m_rank, n}, this->output_buffer.options());
      auto output_to_reduce =
          this->reduce_buffer.slice(0, 0, nnodes * m_rank).view({nnodes, m_rank, this->n_dim});
      bsr_reduce(output_to_reduce, result, tile_M, tile_N);
      return result;
      // return full_output;
    } else {
      int local_world_size = world_size / nnodes;
      if (fuse_reduction) {
        auto length = m / world_size;
        // return this->output_buffer.slice(0, rank * length, (rank + 1) * length).unsqueeze(0);
        return this->output_buffer.slice(0, 0, length).unsqueeze(0);
      } else {
        auto output_4d = full_output.view({nnodes, local_world_size, m / world_size, n});
        auto output = output_4d.sum(1);  // (nnodes,m_rank,n)
        return output;
      }
    }
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    forward_gemm_impl(input, weight, bias, hparams);
    forward_barrier(input, weight, bias);
    return forward_reduce_scatter_impl(input, weight, bias, hparams);
  }

  void
  forward_gemm(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    return forward_gemm_impl(std::move(input), std::move(weight), std::move(bias), c10::nullopt);
  }

  void
  forward_barrier(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    nvshmemx_barrier_all_on_stream(stream);
  }

  torch::Tensor
  forward_reduce_scatter(
      torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    return forward_reduce_scatter_impl(
        std::move(input), std::move(weight), std::move(bias), c10::nullopt);
  }

  torch::Tensor
  forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    return forward_impl(std::move(input), std::move(weight), std::move(bias), c10::nullopt);
  }

  void
  zero_buffers() {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    if (this->output_buffer.defined()) {
      this->output_buffer.zero_();
    }
    if (this->barrier_buffer.defined()) {
      this->barrier_buffer.zero_();
    }
    if (this->reduce_buffer.defined()) {
      this->reduce_buffer.zero_();
    }
    nvshmemx_barrier_all_on_stream(stream);
    if (!no_nvlink) {
      c10::cuda::stream_synchronize(stream);
    }
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    auto meta = unify_type(this->get_gemm_meta(/*has_bias=*/bias.has_value()));
    auto rt_conf = this->get_rt_conf(input, weight, bias);
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto filter_hparams = [&](UnifiedGemmHParams const &hparams) { return true; };

    auto elapsed_tensor = torch::empty({}, weight.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          if (not filter_hparams(hparams)) {
            return;
          }
          // filter non-consistent hparams
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          nvshmemx_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(input, weight, bias, hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          nvshmemx_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          elapsed_tensor.copy_(torch::full({}, avg_elapsed));

          nvshmemx_float_max_reduce_on_stream(
              NVSHMEM_TEAM_WORLD,
              static_cast<float *>(reduced_elapsed_tensor.data_ptr()),
              static_cast<float const *>(elapsed_tensor.data_ptr()),
              1,
              stream);

          float reduce_elapsed = reduced_elapsed_tensor.item().toFloat();
          ctx->add(meta, rt_conf, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);
    return this->forward_impl(
        std::move(input), std::move(weight), std::move(bias), std::move(best_hparams));
  }

  void
  _ensure_topo_initialized() {
    if (!topo_utils::is_topo_initialized()) {
      topo_utils::initialize_topo(const_cast<c10d::ProcessGroup &>(this->tp_group));
    }
  }
};  // namespace flux

namespace py = pybind11;

static int _register_gemm_rs_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_reduce_scatter", [](py::module &m) {
    py::class_<GemmRS>(m, "GemmRS")
        .def(
            py::init([](c10d::ProcessGroup tp_group,
                        int32_t nnodes,
                        int32_t max_m,
                        int32_t n_dim,
                        py::object py_input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight,
                        bool fuse_reduction) {
              auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);

              return new GemmRS(
                  tp_group,
                  nnodes,
                  max_m,
                  n_dim,
                  input_dtype,
                  output_dtype,
                  transpose_weight,
                  fuse_reduction);
            }),
            py::arg("tp_group"),
            py::arg("nnodes"),
            py::arg("max_m"),
            py::arg("n_dim"),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false,
            py::arg("fuse_reduction") = false)
        .def("zero_buffers", &GemmRS::zero_buffers)
        .def(
            "forward",
            &GemmRS::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "forward_gemm",
            &GemmRS::forward_gemm,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "forward_barrier",
            &GemmRS::forward_barrier,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "forward_reduce_scatter",
            &GemmRS::forward_reduce_scatter,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none())
        .def(
            "profiling",
            &GemmRS::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
