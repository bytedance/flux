//===- test_blockscale_gemm_comm_none.cu -------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include <iostream>
#include <fstream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"

// Includes from examples directory

#include "./gemm_with_groupwise_scaling.h"

#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/op_registry.h"
#include "flux/args/comm_none.h"

using namespace cute;
using namespace bytedance::flux;

// Command line options parsing
template <typename RasterOrderOptions>
struct Options {
  bool help = false;

  float alpha = 1.f, beta = 0.f;
  float scale_a = 1.f, scale_b = 1.f, scale_c = 1.f, scale_d = 1.f, scale_aux = 1.f;
  bool device_scale = false;
  bool save_aux = true;
  bool save_amax = true;
  int iterations = 1000;
  int m = 1024, n = 512, k = 1024, l = 1;
  RasterOrderOptions raster;
  int swizzle;

  // Parses the command line
  void
  parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("scale_a", scale_a, 1.f);
    cmd.get_cmd_line_argument("scale_b", scale_b, 1.f);
    cmd.get_cmd_line_argument("scale_c", scale_c, 1.f);
    cmd.get_cmd_line_argument("scale_d", scale_d, 1.f);
    cmd.get_cmd_line_argument("scale_aux", scale_aux, 1.f);
    cmd.get_cmd_line_argument("device_scale", device_scale, false);
    cmd.get_cmd_line_argument("save_aux", save_aux, true);
    cmd.get_cmd_line_argument("save_amax", save_amax, true);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    } else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    } else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);
  }

  /// Prints the usage statement.
  std::ostream &
  print_usage(std::ostream &out) const {
    out << "54_fp8_hopper_warp_specialized_gemm\n\n"
        << "  Hopper FP8 GEMM using a Warp Specialized kernel.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --l=<int>                   Sets the l extent (batch) of the GEMM\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n"
        << "  --scale_a=<f32>             Scaling factor for A\n"
        << "  --scale_b=<f32>             Scaling factor for B\n"
        << "  --scale_c=<f32>             Scaling factor for C\n"
        << "  --scale_d=<f32>             Scaling factor for D (ignored for non-fp8 D)\n"
        << "  --scale_aux=<f32>           Scaling factor for the auxiliary tensor (ignored for "
           "non-fp8 aux)\n"
        << "  --device_scale=<bool>       Copy scalars to device memory before kernel launch "
           "(default: false)\n"
        << "  --save_aux=<bool>           Save the pre-activation as an auxiliary tensor "
           "(default: true)\n"
        << "  --save_amax=<bool>          Save the pre-scaled max absolute value of any fp8 "
           "outputs (aux and/or D) (default: true)\n"
        << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along "
           "M, and H for heuristic)\n\n"
        << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
        << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
        << "$ " << "54_fp8_hopper_warp_specialized_gemm"
        << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double
  gflops(double runtime_s) const {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using ElementA = cutlass::float_e4m3_t;     // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                  // matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB = cutlass::float_e4m3_t;        // Element type for B matrix operand
using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                  // matrix in units of elements (up to 16 bytes)

// C matrix configuration
using ElementC = cutlass::float_e4m3_t;        // Element type for C and D matrix operands
using LayoutC = cutlass::layout::ColumnMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C
                                                  // matrix in units of elements (up to 16 bytes)

// D matrix configuration
using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

// Auxiliary matrix configuration and other fusion types
using ElementAux = ElementC;
using LayoutAux = LayoutC;
using ElementAmax = float;
using ElementBias = float;

// Core kernel configurations
using ElementAccumulator = float;  // Element type for internal accumulation
using ElementBlockScale = float;   // Element type for blockscaling during accumulation
using ElementCompute = float;      // Element type for epilogue computation
using ArchTag =
    cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
using TileShape = Shape<_128, _128, _128>;             // Threadblock-level tile size
using ClusterShape = Shape<_1, _2, _1>;                // Shape of the threadblocks in a cluster

constexpr int ScaleMsPerTile = 128;
constexpr int ScaleGranularityM = size<0>(TileShape{}) / ScaleMsPerTile;

using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<ScaleGranularityM>;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using FusionOperation = cutlass::epilogue::fusion::ScaledLinCombPerRowBiasEltActAmaxAux<
    LayoutAux,
    cutlass::epilogue::thread::Identity,
    ElementD,
    ElementCompute,
    ElementAux,
    ElementAmax,
    ElementBias,
    ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    TileShape,
    ClusterShape,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementD,
    LayoutD,
    AlignmentD,
    EpilogueSchedule,
    FusionOperation>::CollectiveOp;

using CollectiveMainloopWithBlockWiseScaling =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        LayoutA,
        AlignmentA,
        ElementB,
        LayoutB,
        AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,  // Indicates ProblemShape
    CollectiveMainloopWithBlockWiseScaling,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Extract information from Gemm kernel.
using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
using ElementScalar = typename EpilogueOutputOp::ElementScalar;
using ElementAmax = typename EpilogueOutputOp::ElementAmax;
using ActivationFunctor = typename EpilogueOutputOp::ActivationFn;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using StrideAux = StrideD;

constexpr bool IsDFp8 = cute::is_same_v<ElementD, cutlass::float_e4m3_t> or
                        cute::is_same_v<ElementD, cutlass::float_e5m2_t>;

constexpr bool IsAuxFp8 = cute::is_same_v<ElementAux, cutlass::float_e4m3_t> or
                          cute::is_same_v<ElementAux, cutlass::float_e5m2_t>;

static_assert(
    size<0>(TileShape{}) == ScaleGranularityM * ScaleMsPerTile,
    "FP8 scaling granularity must evenly divide tile shape along M.");

static_assert(
    cute::is_same_v<ElementAccumulator, ElementBlockScale>,
    "ElementAccumulator and ElementBlockScale should be same datatype");

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
StrideAux stride_aux;
uint64_t seed;

cutlass::HostTensor<ElementA, LayoutA> tensor_A;
cutlass::HostTensor<ElementB, LayoutB> tensor_B;
cutlass::HostTensor<ElementC, LayoutC> tensor_C;
cutlass::HostTensor<ElementD, LayoutD> tensor_D;
uint32_t mma_promotion_interval;
cutlass::HostTensor<ElementBlockScale, LayoutA> blockscale_tensor_A;
cutlass::HostTensor<ElementBlockScale, LayoutB> blockscale_tensor_B;
cutlass::HostTensor<ElementD, LayoutD> tensor_ref_D;
cutlass::HostTensor<ElementAux, LayoutAux> tensor_aux;
cutlass::HostTensor<ElementAux, LayoutAux> tensor_ref_aux;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_alpha;
cutlass::HostTensor<ElementScalar, LayoutScalar> scalar_beta;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_A;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_B;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_C;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_D;
cutlass::HostTensor<ElementScalar, LayoutScalar> scale_aux;
cutlass::HostTensor<ElementAmax, LayoutScalar> abs_max_D;
cutlass::HostTensor<ElementAmax, LayoutScalar> reference_abs_max_D;
cutlass::HostTensor<ElementAmax, LayoutScalar> abs_max_aux;
cutlass::HostTensor<ElementAmax, LayoutScalar> reference_abs_max_aux;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions =
    typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

/// Result structure
struct Result {
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
      double avg_runtime_ms = 0,
      double gflops = 0,
      cutlass::Status status = cutlass::Status::kSuccess,
      cudaError_t error = cudaSuccess)
      : avg_runtime_ms(avg_runtime_ms),
        gflops(gflops),
        status(status),
        error(error),
        passed(false) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool
initialize_tensor(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  double scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;
  int bits_output = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else if (bits_output == 16) {
    scope_max = 5;
    scope_min = -5;
  } else {
    scope_max = 8;
    scope_min = -8;
  }

  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);

  return true;
}

/// Helper to initialize a block of device data (scale_tensors)
template <typename Element, typename Layout>
bool
initialize_scale_tensor(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
  double scope_max, scope_min;

  scope_min = -1;
  scope_max = 1;

  cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void
initialize(const Options<RasterOrderOptions> &options) {
  // Find Group Scaling tensor shapes based on `ScaleGranularityM`, problem shape, and TileShape
  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);
  auto blockscale_shape =
      shape(get<1>(cute::zipped_divide(cute::make_layout(gemm_problem_shape), TileShape{})));
  auto groupscale_m =
      cute::get<0>(blockscale_shape) * ScaleMsPerTile;  // We need to pad along M in scale tensor
                                                        // of A to prevent illegal memory access.
  auto blockscale_n = cute::get<1>(blockscale_shape);
  auto blockscale_k = cute::get<2>(blockscale_shape);

  stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(options.m, options.n, options.l));
  stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(options.m, options.n, options.l));
  stride_aux = stride_D;

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto groupscale_a_coord = cutlass::make_Coord(groupscale_m * options.l, blockscale_k);
  auto blockscale_b_coord = cutlass::make_Coord(blockscale_k, blockscale_n * options.l);

  tensor_A.resize(a_coord);
  blockscale_tensor_A.resize(groupscale_a_coord);
  tensor_B.resize(b_coord);
  blockscale_tensor_B.resize(blockscale_b_coord);
  tensor_C.resize(c_coord);
  tensor_D.resize(c_coord);
  tensor_ref_D.resize(c_coord);

  initialize_tensor(tensor_A.host_view(), seed + 2022);
  initialize_tensor(tensor_B.host_view(), seed + 2023);
  initialize_tensor(tensor_C.host_view(), seed + 2024);
  initialize_scale_tensor(blockscale_tensor_A.host_view(), seed + 2025);
  initialize_scale_tensor(blockscale_tensor_B.host_view(), seed + 2026);

#if 0  // Dump blockscaled tensors
  std::cout << "blockscale_tensor_A: " << groupscale_a_coord << std::endl;
  std::cout << blockscale_tensor_A.host_view() << "\n";
  std::cout << "blockscale_tensor_B: " << blockscale_b_coord << std::endl;
  std::cout << blockscale_tensor_B.host_view() << "\n";
#endif

  // Print group scaling tensors on the host side.
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();
  blockscale_tensor_A.sync_device();
  blockscale_tensor_B.sync_device();

  mma_promotion_interval = 4;

  if (options.save_aux) {
    tensor_aux.resize(c_coord);
    tensor_aux.sync_device();
    tensor_ref_aux.resize(c_coord);
  }

  if (options.device_scale) {
    scalar_alpha.resize(cutlass::make_Coord(1));
    scalar_beta.resize(cutlass::make_Coord(1));
    scale_A.resize(cutlass::make_Coord(1));
    scale_B.resize(cutlass::make_Coord(1));
    scale_C.resize(cutlass::make_Coord(1));
    scale_D.resize(cutlass::make_Coord(1));
    scale_aux.resize(cutlass::make_Coord(1));

    cutlass::reference::host::TensorFill(scalar_alpha.host_view(), options.alpha);
    cutlass::reference::host::TensorFill(scalar_beta.host_view(), options.beta);
    cutlass::reference::host::TensorFill(scale_A.host_view(), options.scale_a);
    cutlass::reference::host::TensorFill(scale_B.host_view(), options.scale_b);
    cutlass::reference::host::TensorFill(scale_C.host_view(), options.scale_c);
    cutlass::reference::host::TensorFill(scale_D.host_view(), options.scale_d);
    cutlass::reference::host::TensorFill(scale_aux.host_view(), options.scale_aux);

    scalar_alpha.sync_device();
    scalar_beta.sync_device();
    scale_A.sync_device();
    scale_B.sync_device();
    scale_C.sync_device();
    scale_D.sync_device();
    scale_aux.sync_device();
  }

  if (IsDFp8 && options.save_amax) {
    abs_max_D.resize(cutlass::make_Coord(1));
    abs_max_D.sync_device();
    reference_abs_max_D.resize(cutlass::make_Coord(1));
  }

  if (IsAuxFp8 && options.save_aux && options.save_amax) {
    abs_max_aux.resize(cutlass::make_Coord(1));
    abs_max_aux.sync_device();
    reference_abs_max_aux.resize(cutlass::make_Coord(1));
  }
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments
args_from_options(const Options<RasterOrderOptions> &options) {
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {tensor_A.device_data(),
       stride_A,
       tensor_B.device_data(),
       stride_B,
       mma_promotion_interval,
       blockscale_tensor_A.device_data(),
       blockscale_tensor_B.device_data()},
      {{},  // epilogue.thread
       tensor_C.device_data(),
       stride_C,
       tensor_D.device_data(),
       stride_D}};

  auto &fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = options.alpha;
  fusion_args.beta = options.beta;
  fusion_args.alpha_ptr = scalar_alpha.device_data();
  fusion_args.beta_ptr = scalar_beta.device_data();
  fusion_args.scale_a = options.scale_a;
  fusion_args.scale_b = options.scale_b;
  fusion_args.scale_c = options.scale_c;
  fusion_args.scale_a_ptr = scale_A.device_data();
  fusion_args.scale_b_ptr = scale_B.device_data();
  fusion_args.scale_c_ptr = scale_C.device_data();

  // ignored if tensor types are not fp8
  fusion_args.scale_d = options.scale_d;
  fusion_args.scale_aux = options.scale_aux;
  fusion_args.scale_d_ptr = scale_D.device_data();
  fusion_args.scale_aux_ptr = scale_aux.device_data();

  // leaving/setting these as nullptr disables the fusion at runtime
  fusion_args.bias_ptr = nullptr;

  if (options.save_aux) {
    fusion_args.aux_ptr = tensor_aux.device_data();
    fusion_args.dAux = stride_aux;
    if (options.save_amax) {
      fusion_args.amax_aux_ptr = abs_max_aux.device_data();
    }
  }

  if (options.save_amax) {
    fusion_args.amax_D_ptr = abs_max_D.device_data();
  }

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and
  // 8)
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

bool
verify(const Options<RasterOrderOptions> &options) {
  //
  // Compute reference output
  //

  // Group scaling tensors shapes based `ScaleGranularityM`, CTA Block (TileShape) and GEMM Problem
  // shape
  auto gemm_problem_shape = cute::make_shape(options.m, options.n, options.k);
  auto blockscale_shape =
      shape(get<1>(cute::zipped_divide(cute::make_layout(gemm_problem_shape), TileShape{})));
  auto blockscale_m = cute::get<0>(blockscale_shape);
  auto blockscale_n = cute::get<1>(blockscale_shape);
  auto blockscale_k = cute::get<2>(blockscale_shape);
#if 0
  std::cout << "gemm problem shape: " << gemm_problem_shape << std::endl;
  std::cout << "gemm blockscale_shape: " << blockscale_shape << std::endl;
  std::cout << "gemm blockscale_m: " << blockscale_m << std::endl;
  std::cout << "gemm blockscale_n: " << blockscale_n << std::endl;
  std::cout << "gemm blockscale_k: " << blockscale_k << std::endl;
#endif
  // Create instantiation for device reference gemm kernel
  auto A = cute::make_tensor(
      tensor_A.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto B = cute::make_tensor(
      tensor_B.host_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto C = cute::make_tensor(
      tensor_C.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_C));
  auto D = cute::make_tensor(
      tensor_ref_D.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_D));
  auto Aux = cute::make_tensor(
      tensor_ref_aux.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_aux));

  auto blockscale_A = cute::make_tensor(
      blockscale_tensor_A.host_data(),
      cute::make_layout(
          cute::make_shape(blockscale_m, ScaleMsPerTile, blockscale_k, options.l),
          cute::make_stride(
              blockscale_k * ScaleMsPerTile,
              1,
              ScaleMsPerTile,
              blockscale_m * blockscale_k * ScaleMsPerTile)));
  auto blockscale_B = cute::make_tensor(
      blockscale_tensor_B.host_data(),
      cute::make_layout(
          cute::make_shape(blockscale_n, blockscale_k, options.l),
          cute::make_stride(blockscale_k, 1, blockscale_n * blockscale_k)));
#if 0
  std::cout << "A: " << std::endl;
  cute::print(A.shape());
  printf("\n");

  std::cout << "B: " << std::endl;
  cute::print(B.shape());
  printf("\n");

  std::cout << "C: " << std::endl;
  cute::print(C.shape());
  printf("\n");

  std::cout << "D: " << std::endl;
  cute::print(D.shape());
  printf("\n");

  std::cout << "block scale A: " << std::endl;
  cute::print(blockscale_A.shape());
  printf("\n");

  std::cout << "block scale B: " << std::endl;
  cute::print(blockscale_B.shape());
  printf("\n");
#endif
  using unused_t = decltype(D);

  cutlass::reference::host::GettMainloopParams<
      ElementAccumulator,
      decltype(A),
      decltype(B),
      decltype(blockscale_A),
      decltype(blockscale_B),
      TileShape>
      mainloop_params{
          A,
          B,  // Operand Tensors
          blockscale_A,
          blockscale_B  // Groupwise scaling Tensors
      };

  cutlass::reference::host::GettEpilogueParams<
      ElementScalar,
      ElementScalar,
      ElementAccumulator,
      ElementCompute,
      decltype(C),
      decltype(D),
      unused_t,  // bias
      decltype(Aux),
      unused_t,  // valpha
      unused_t,  // vbeta
      ActivationFunctor>
      epilogue_params;

  epilogue_params.C = C;
  epilogue_params.D = D;
  epilogue_params.Aux = Aux;
  epilogue_params.alpha = options.alpha;
  epilogue_params.beta = options.beta;
  epilogue_params.scale_a = options.scale_a;
  epilogue_params.scale_b = options.scale_b;
  epilogue_params.scale_c = options.scale_c;
  epilogue_params.scale_d = options.scale_d;
  epilogue_params.scale_aux = options.scale_aux;
  epilogue_params.abs_max_D = reference_abs_max_D.host_data();
  epilogue_params.abs_max_Aux = reference_abs_max_aux.host_data();

  // get reference result
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // compare_reference
  tensor_D.sync_host();
  bool passed =
      cutlass::reference::host::TensorEquals(tensor_ref_D.host_view(), tensor_D.host_view());

  if (false) {
    std::cout << "tensor_ref_D.host_view() {" << std::endl
              << tensor_ref_D.host_view() << std::endl
              << "}" << std::endl;
    std::cout << "tensor_D.host_view() {" << std::endl
              << tensor_D.host_view() << std::endl
              << "}" << std::endl;
  }

  if (IsDFp8 && options.save_amax) {
    abs_max_D.sync_host();
    passed &=
        abs_max_D.at(cutlass::make_Coord(0)) == reference_abs_max_D.at(cutlass::make_Coord(0));
  }

  if (options.save_aux) {
    tensor_aux.sync_host();
    passed &=
        cutlass::reference::host::TensorEquals(tensor_ref_aux.host_view(), tensor_aux.host_view());
    if (IsAuxFp8 && options.save_amax) {
      abs_max_aux.sync_host();
      passed &= abs_max_aux.at(cutlass::make_Coord(0)) ==
                reference_abs_max_aux.at(cutlass::make_Coord(0));
    }
  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int
run(Options<RasterOrderOptions> &options) {
  initialize(options);

  // ===== Get Op from Registry, and Profile =====
  printf("== Flux Registered Op ==\n");

  using DType = decltype(make_gemm_dtype_config(_E4M3{}, _E4M3{}, _E4M3{}, _E4M3{}));

  auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(DType{}));
  // UnifiedImplMeta impl_spec = None{};
  // impl_spec = make_gemm_v3_meta(use_fast_accum, false);
  auto meta = make_gemm_meta(
      dt_conf, get_arch(), _CommNone{}, _RCC{}, _GemmV3{}(), make_gemm_v3_meta(false, true));

  auto rt_conf = make_runtime_config(options.m, options.n, options.k);
  auto hparams = OpRegistry::instance().get_hparams(meta, rt_conf);
  auto gemm_op = OpRegistry::instance().get_op(meta, hparams);

  auto stream = nullptr;
  BlockScaleGemmArguments args = BlockScaleGemmArguments{
      .m = options.m,
      .n = options.n,
      .k = options.k,
      .l = options.l,
      .A = tensor_A.device_data(),
      .B = tensor_B.device_data(),
      .mma_promotion_interval = 4,
      .blockscale_A = blockscale_tensor_A.device_data(),
      .blockscale_B = blockscale_tensor_B.device_data(),
      .C = tensor_C.device_data(),
      .D = tensor_D.device_data(),
      .alpha = options.alpha,
      .beta = options.beta,
      .scale_a = options.scale_a,
      .scale_b = options.scale_b,
      .scale_c = options.scale_c,
      .scale_d = options.scale_d,
      .scale_aux = options.scale_aux,
      .bias = nullptr,
  };

  int64_t flux_workspace_size = gemm_op->get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> flux_workspace(flux_workspace_size);
  // cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result_flux;
  gemm_op->run(args, flux_workspace.get(), stream);

  result_flux.passed = verify(options);
  std::cout << "  Disposition: " << (result_flux.passed ? "Passed" : "Failed") << std::endl;

  constexpr int warm_iters = 5;
  int iters = options.iterations;
  GpuTimer timer;
  for (int i = 0; i < warm_iters + iters; ++i) {
    if (i == warm_iters) {
      timer.start(stream);
    }
    gemm_op->run(args, flux_workspace.get(), stream);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();

  // Compute average runtime and GFLOPs.
  result_flux.avg_runtime_ms = double(timer.elapsed_millis()) / double(options.iterations);
  result_flux.gflops = options.gflops(result_flux.avg_runtime_ms / 1000.0);
  std::cout << "  Avg runtime: " << result_flux.avg_runtime_ms << " ms" << std::endl;
  std::cout << "  GFLOPS: " << result_flux.gflops << std::endl;

  // ===== End: Test Op Registry =====
  printf("== CUTLASS GEMM Kernel ==\n");

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  // Run profiling loop
  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    } else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x'
              << options.l << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of "
              << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return 0;
}

int
main(int argc, char const **args) {
  // Parse options
  Options<RasterOrderOptions> options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // Evaluate Flux and CUTLASS kernels
  run<Gemm>(options);

  return 0;
}