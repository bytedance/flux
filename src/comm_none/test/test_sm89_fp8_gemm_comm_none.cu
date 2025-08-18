//===- test_sm89_fp8_gemm_comm_none.cu ---------------------------- C++ ---===//
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

#include <fstream>
#include <cstddef>
#include <type_traits>
#include "cute/stride.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/arch/mma.h"
#include "cutlass/bfloat16.h"
#include "cutlass/coord.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_size.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/op_registry.h"
#include "flux/args/comm_none.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"
#include "cutlass/core_io.h"
#include "cutlass/util/tensor_view_io.h"
#if defined(WITH_GFLAGS)
#include <gflags/gflags.h>
#endif

using namespace cute;
using namespace bytedance::flux;
using ElementAbsmax = cutlass::bfloat16_t;

// Command line options parsing
struct Options {
  bool help;
  bool error;
  bool reference_check;
  cutlass::gemm::GemmCoord problem_size;

  int iterations;
  int warmup_iterations;

  int dtype_A;
  int dtype_C;
  int gemm_layout;

  bool scale_A;
  bool scale_B;
  bool scale_C;

  float alpha;
  float beta;

  Options()
      : help(false),
        error(false),
        reference_check(false),
        iterations(20),
        warmup_iterations(5),
        dtype_A((int)DataTypeEnum::E4M3),
        dtype_C((int)DataTypeEnum::E4M3),
        gemm_layout((int)GemmLayoutEnum::RCR),
        scale_A(true),
        scale_B(true),
        scale_C(true),
        alpha(1.f),
        beta(0.f) {}

  // Parses the command line
  void
  parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, 5);
    cmd.get_cmd_line_argument("reference-check", reference_check, false);
    cmd.get_cmd_line_argument("scale-A", scale_A, true);
    cmd.get_cmd_line_argument("scale-B", scale_B, true);
    cmd.get_cmd_line_argument("scale-C", scale_C, true);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);

    // 0 for Void, 1 for FP16, 2 for BF16, 3 for FP32, 4 for E4M3, 5 for E5M2
    cmd.get_cmd_line_argument("dtype_A", dtype_A, 4);
    cmd.get_cmd_line_argument("dtype_C", dtype_C, 2);
    cmd.get_cmd_line_argument("gemm_layout", gemm_layout, 1);

    int m, n, k;
    cmd.get_cmd_line_argument("m", m, 1024);
    cmd.get_cmd_line_argument("n", n, 1024);
    cmd.get_cmd_line_argument("k", k, 1024);

    problem_size = cutlass::gemm::GemmCoord{m, n, k};
  }

  /// Prints the usage statement.
  std::ostream &
  print_usage(std::ostream &out) const {
    out << "58_ada_fp8_gemm\n\n"
        << "  This example executes a GEMM using Ada FP8 Tensor Core operations. In addition to "
           "performing\n"
        << "  a normal GEMM, the kernel performs the following operations:\n"
        << "      Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) "
           "+ bias\n"
        << "        D = activation(Aux)\n\n"
        << "      if Aux is fp8:\n"
        << "         abs_max_output = max( abs(aux) | (for every aux in Aux) )\n"
        << "         Aux = scale_aux * Aux\n\n"
        << "      if D is fp8 type:\n"
        << "         abs_max_output = max( abs(d) | (for every d in D) )\n"
        << "         D = scale_d * D\n\n"
        << "Options:\n\n"
        << "  --help                           If specified, displays this usage statement\n\n"
        << "  --m=<int>                        Sets the M dimension of the GEMM\n"
        << "  --n=<int>                        Sets the N dimension of the GEMM\n"
        << "  --k=<int>                        Sets the K dimension of the GEMM\n"
        << "  --scale-A=<bool>                 Whether to apply a scaling factor to operand A "
           "(default: true)\n"
        << "  --scale-B=<bool>                 Whether to apply a scaling factor to operand B "
           "(default: true)\n"
        << "  --scale-C=<bool>                 Whether to apply a scaling factor to operand C "
           "(default: true)\n"
        << "  --dtype_A=<int>                  4 for E4M3, 5 for E5M2. dtype_B is the same as "
           "dtype_A(default: 4)\n"
        << "  --dtype_C=<int>                  0 for void, 2 for BF16, 5 for E5M2. (default: 2)\n"
        << "  --gemm_layout=<int>              0 for RRR, 1 for RCR, 2 for RCC. (default: 1)\n"
        << "  --iterations=<int>               Number of profiling iterations to perform\n"
        << "  --warmup-iterations=<int>        Number of warmup iterations to perform\n"
        << "  --reference-check=<bool>         If true, performs reference check\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  float
  gflops(float runtime_s) const {
    // Two flops per multiply-add
    return 2.0f * float(problem_size.product()) / float(1.0e9) / runtime_s;
  }
};

/// Helper class to run the kernel
template <typename DType, GemmLayoutEnum layout = GemmLayoutEnum::RCR, bool kUseFastAccum = false>
struct TestbedRunner {
  constexpr static auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(DType{}));
  using ElementA = decltype(to_cutlass_element(dt_conf.a()));
  using ElementB = decltype(to_cutlass_element(dt_conf.b()));
  using ElementOutput = decltype(to_cutlass_element(dt_conf.d()));
  using ElementAuxOutput = ElementOutput;
  using ElementAccumulator = float;
  using LayoutA = decltype(to_cutlass_layout_a(cute::C<layout>{}));
  using LayoutB = decltype(to_cutlass_layout_b(cute::C<layout>{}));
  using LayoutC = decltype(to_cutlass_layout_c(cute::C<layout>{}));
  static int const kStages = 3;
  static int const kAlignmentA = 16;
  static int const kAlignmentB = 16;

  static int const Count = cute::min(8, 128 / cutlass::sizeof_bits<ElementOutput>::value);
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
      cutlass::epilogue::thread::Identity,
      ElementOutput,
      ElementAuxOutput,
      Count,
      ElementAccumulator,
      ElementAccumulator>;

  template <typename MathOperator>
  using Gemm_ = cutlass::gemm::device::GemmUniversalWithAbsMax<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementOutput,
      LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm89,
      cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 32>,
      EpilogueOutputOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      kStages,
      kAlignmentA,
      kAlignmentB,
      MathOperator>;
  using Gemm = conditional_t<
      kUseFastAccum,
      Gemm_<cutlass::arch::OpMultiplyAddFastAccum>,
      Gemm_<cutlass::arch::OpMultiplyAdd>>;

  using ElementAbsmax = typename EpilogueOutputOp::ElementAbsmax;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;
  using ElementScalingFactor = typename Gemm::EpilogueOutputOp::ElementScalingFactor;

  static bool const kScaleAux = Gemm::EpilogueOutputOp::kIsScalingAndAmaxAuxOutputNeeded;
  static bool const kScaleOutput = Gemm::EpilogueOutputOp::kIsScalingAndAmaxOutputNeeded;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC>
      tensor_Aux;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC>
      tensor_D;
  cutlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_Vector;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC>
      reference1_D;
  cutlass::HostTensor<ElementAccumulator, typename Gemm::LayoutC> tmp_D;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC>
      reference2_D;
  cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC>
      reference_Aux;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_A;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_B;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_C;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_D;
  cutlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> abs_max_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> abs_max_D;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> reference_abs_max_Aux;
  cutlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> reference_abs_max_D;

  //
  // Methods
  //

  TestbedRunner(
      bool scaleA = true,
      bool scaleB = true,
      bool scaleC = true,
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      uint64_t seed_ = 2080)
      : init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) {}

  /// Helper to initialize scaling factors
  template <typename Element, typename Layout>
  bool
  initialize_scale_factor(cutlass::TensorView<Element, Layout> view, uint64_t seed, int bits = 0) {
    cutlass::reference::host::TensorFillRandomUniform(view, seed, double(1.), double(0.), bits);
    return true;
  }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool
  initialize_tensor(
      cutlass::TensorView<Element, Layout> view,
      cutlass::Distribution::Kind dist_kind,
      uint64_t seed) {
    if (dist_kind == cutlass::Distribution::Uniform) {
      double scope_max, scope_min;
      int bits_input = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Gemm::ElementC>::value;

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
    } else if (dist_kind == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(view);
    } else if (dist_kind == cutlass::Distribution::Gaussian) {
      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    } else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    } else {
      std::cerr << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void
  initialize(const Options &options) {
    //
    // Allocate the GEMM workspace
    //

    tensor_A.resize(options.problem_size.mk());
    tensor_B.resize(options.problem_size.kn());
    tensor_C.resize(options.problem_size.mn());
    tensor_D.resize(options.problem_size.mn());
    tensor_Vector.resize({1, options.problem_size.n()});
    reference1_D.resize(options.problem_size.mn());
    reference2_D.resize(options.problem_size.mn(), false);
    tmp_D.resize(options.problem_size.mn(), false);

    initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
    initialize_tensor(tensor_B.host_view(), init_B, seed + 2018);
    initialize_tensor(tensor_C.host_view(), init_C, seed + 2017);
    initialize_tensor(tensor_Vector.host_view(), init_C, seed + 2020);

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    cutlass::Coord<2> origin(0);
    tensor_A.host_view().at(origin) = typename Gemm::ElementA(1);
    tensor_B.host_view().at(origin) = typename Gemm::ElementB(1);
    tensor_C.host_view().at(origin) = typename Gemm::ElementC(1);
    tensor_Vector.host_view().at(origin) = typename Gemm::ElementC(1);

    cutlass::reference::host::TensorFill(tensor_D.host_view());
    cutlass::reference::host::TensorCopy(reference1_D.host_view(), tensor_C.host_view());
    cutlass::reference::host::TensorCopy(reference2_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    tensor_Vector.sync_device();

    int scale_bits = 2;
    if (options.scale_A) {
      scale_A.resize({1, 1});
      initialize_scale_factor(scale_A.host_view(), seed + 2021, scale_bits);
      scale_A.sync_device();
    }

    if (options.scale_B) {
      scale_B.resize({1, 1});
      initialize_scale_factor(scale_B.host_view(), seed + 2022, scale_bits);
      scale_B.sync_device();
    }

    if (options.scale_C) {
      scale_C.resize({1, 1});
      initialize_scale_factor(scale_C.host_view(), seed + 2023, scale_bits);
      scale_C.sync_device();
    }

    if (kScaleOutput) {
      scale_D.resize({1, 1});
      initialize_scale_factor(scale_D.host_view(), seed + 2024, scale_bits);
      scale_D.sync_device();

      abs_max_D.resize({1, 1});
      cutlass::reference::host::TensorFill(abs_max_D.host_view());
      abs_max_D.sync_device();

      reference_abs_max_D.resize({1, 1});
    }

    if (kScaleAux) {
      tensor_Aux.resize(options.problem_size.mn());
      cutlass::reference::host::TensorFill(tensor_Aux.host_view());
      tensor_Aux.sync_device();

      scale_Aux.resize({1, 1});
      initialize_scale_factor(scale_Aux.host_view(), seed + 2025, scale_bits);
      scale_Aux.sync_device();

      abs_max_Aux.resize({1, 1});
      cutlass::reference::host::TensorFill(abs_max_Aux.host_view());
      abs_max_Aux.sync_device();

      reference_Aux.resize(options.problem_size.mn());
      reference_abs_max_Aux.resize({1, 1});
    }
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool
  compare_reference(const Options &options) {
    tensor_D.sync_host();
    reference1_D.sync_host();

    bool passed =
        cutlass::reference::host::TensorEquals(reference1_D.host_view(), tensor_D.host_view());
    passed &=
        cutlass::reference::host::TensorEquals(reference2_D.host_view(), tensor_D.host_view());

    if (kScaleAux) {
      tensor_Aux.sync_host();
      abs_max_Aux.sync_host();
      reference_Aux.sync_host();
      reference_abs_max_Aux.sync_host();
      passed &= cutlass::reference::host::TensorEquals(
          reference_Aux.host_view(), tensor_Aux.host_view());
      if (passed) {
        std::cout << "abs_max_Aux passed\n";
      } else {
        std::cout << "abs_max_Aux failed\n";
      }
      passed &= cutlass::reference::host::TensorEquals(
          abs_max_Aux.host_view(), reference_abs_max_Aux.host_view());
      if (passed) {
        std::cout << "abs_max_Aux passed\n";
      } else {
        std::cout << "abs_max_Aux failed\n";
      }
    }

    if (kScaleOutput) {
      abs_max_D.sync_host();
      reference_abs_max_D.sync_host();
      passed &= cutlass::reference::host::TensorEquals(
          abs_max_D.host_view(), reference_abs_max_D.host_view());
    }

    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;

      std::string output_file = "testbed_with_amax_errors.txt";
      std::ofstream file(output_file);

      file << "problem: " << options.problem_size << ", alpha: " << options.alpha
           << ", beta: " << options.beta << "\n\n";

      file << "A =\n"
           << tensor_A.host_view() << "\nB =\n"
           << tensor_B.host_view() << "\nC =\n"
           << tensor_C.host_view() << "\nVector =\n"
           << tensor_Vector.host_view() << "\nScaleA = " << scale_A.host_view()
           << "\nScaleB = " << scale_B.host_view() << "\nScaleC = " << scale_C.host_view()
           << "\nScaleD = " << scale_D.host_view() << "\nScaleAux = " << scale_Aux.host_view()
           << "\n\nReference1 D =\n"
           << reference1_D.host_view() << "\nReference2 D =\n"
           << reference2_D.host_view() << "\nComputed D =\n"
           << tensor_D.host_view();
      if (kScaleAux) {
        file << "\n\nReference Aux =\n"
             << reference_Aux.host_view() << "\nComputed Aux =\n"
             << tensor_Aux.host_view()
             << "\n\nReference Absmax Aux = " << reference_abs_max_Aux.host_view()
             << "\nComputed Absmax Aux = " << abs_max_Aux.host_view();
      }
      if (kScaleOutput) {
        file << "\n\nReference Absmax D = " << reference_abs_max_D.host_view()
             << "\nComputed Absmax D = " << abs_max_D.host_view();
      }

      std::cerr << "Dumped results to " << output_file << std::endl;
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool
  verify(const Options &options) {
    //
    // Initialize the GEMM operator
    //

    typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{
        ElementCompute(options.alpha), ElementCompute(options.beta)};
    typename Gemm::EpilogueOutputOp::Params epilogue_params{
        activation_params,
        scale_A.device_data(),
        scale_B.device_data(),
        scale_C.device_data(),
        scale_D.device_data(),
        scale_Aux.device_data(),
        reference_abs_max_Aux.device_data(),
        reference_abs_max_D.device_data()};

    constexpr bool is_row_major_A = std::is_same_v<LayoutA, cutlass::layout::RowMajor>;
    static_assert(is_row_major_A);
    constexpr bool is_row_major_B = std::is_same_v<LayoutB, cutlass::layout::RowMajor>;
    constexpr bool is_row_major_C = std::is_same_v<LayoutC, cutlass::layout::RowMajor>;

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        options.problem_size,
        /* batch_count = */ 1,
        epilogue_params,
        tensor_A.device_data(),
        tensor_B.device_data(),
        tensor_C.device_data(),
        reference1_D.device_data(),
        reference_Aux.device_data(),
        tensor_Vector.device_data(),
        options.problem_size.m() * options.problem_size.k(),
        options.problem_size.n() * options.problem_size.k(),
        options.problem_size.m() * options.problem_size.n(),
        options.problem_size.m() * options.problem_size.n(),
        (int)options.problem_size.m(),  // Batch stride vector
        options.problem_size.k(),
        is_row_major_B ? options.problem_size.n() : options.problem_size.k(),
        is_row_major_C ? options.problem_size.n() : options.problem_size.m(),
        is_row_major_C ? options.problem_size.n() : options.problem_size.m(),
        (int64_t)0  // Leading dimension of vector. This must be 0
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    //
    // Run the GEMM
    //

    status = gemm_op();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    cudaError_t cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_error) << std::endl;
      return false;
    }

    /// Reference2: Host reference implementation
    cutlass::Coord<2> origin(0);
    ElementCompute scaled_alpha = options.alpha;
    if (options.scale_A) {
      scaled_alpha *= scale_A.host_view().at(origin);
    }
    if (options.scale_B) {
      scaled_alpha *= scale_B.host_view().at(origin);
    }
    ElementCompute scaled_beta = options.beta;
    if (options.scale_C) {
      scaled_beta *= scale_C.host_view().at(origin);
    }
    cutlass::reference::host::GemmComplex<
        typename Gemm::ElementA,
        typename Gemm::LayoutA,
        typename Gemm::ElementB,
        typename Gemm::LayoutB,
        typename Gemm::ElementC,
        typename Gemm::LayoutC,
        ElementCompute,
        ElementAccumulator,
        ElementAccumulator>(
        options.problem_size,
        scaled_alpha,
        tensor_A.host_ref(),
        Gemm::kTransformA,
        tensor_B.host_ref(),
        Gemm::kTransformB,
        scaled_beta,
        tensor_C.host_ref(),
        tmp_D.host_ref(),
        ElementAccumulator(0));

    ElementCompute tmp_abs_max_Aux(0.);
    ElementCompute tmp_abs_max_D(0.);

    cutlass::NumericConverter<ElementCompute, typename Gemm::ElementC> cvt_c_to_compute;
    cutlass::NumericConverter<ElementCompute, ElementAccumulator> cvt_accum_to_compute;
    cutlass::NumericConverter<ElementAccumulator, ElementCompute> cvt_compute_to_accum;
    cutlass::NumericConverter<typename Gemm::EpilogueOutputOp::ElementOutput, ElementCompute>
        cvt_compute_to_d;
    cutlass::NumericConverter<typename Gemm::EpilogueOutputOp::ElementAuxOutput, ElementCompute>
        cvt_compute_to_aux;

    cutlass::absolute_value_op<ElementCompute> abs;
    cutlass::maximum_with_nan_propogation<ElementCompute> max;
    cutlass::epilogue::thread::Identity<ElementCompute> act;

    ElementScalingFactor d_scale =
        kScaleOutput ? scale_D.host_view().at(origin) : ElementScalingFactor(1.);
    for (int m = 0; m < options.problem_size.m(); ++m) {
      for (int n = 0; n < options.problem_size.n(); ++n) {
        ElementCompute intermediate = cvt_accum_to_compute(tmp_D.host_view().at({m, n}));
        ElementCompute bias = cvt_c_to_compute(tensor_Vector.host_view().at({0, n}));
        ElementCompute aux = intermediate + bias;
        ElementCompute d = act(aux);
        tmp_abs_max_Aux = max(abs(aux), tmp_abs_max_Aux);
        tmp_abs_max_D = max(abs(d), tmp_abs_max_D);
        reference2_D.host_view().at({m, n}) = cvt_compute_to_d(d * d_scale);

        if (kScaleAux) {
          reference_Aux.host_view().at({m, n}) =
              cvt_compute_to_aux(aux * scale_Aux.host_view().at(origin));
        }
      }
    }
    if (kScaleAux) {
      reference_abs_max_Aux.host_view().at(origin) = cvt_compute_to_accum(tmp_abs_max_Aux);
    }
    if (kScaleOutput) {
      reference_abs_max_D.host_view().at(origin) = cvt_compute_to_accum(tmp_abs_max_D);
    }

    return compare_reference(options);
  }

  /// Returns true if the CUDA device is sufficient to execute the kernel.
  bool
  sufficient() const {
    if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4)) {
      std::cerr << "This example requires CUDA 12.4 or greater." << std::endl;
      return false;
    }

    size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);

    cudaDeviceProp properties;
    int device_idx;
    cudaError_t result = cudaGetDevice(&device_idx);

    if (result != cudaSuccess) {
      std::cerr << "cudaGetDevice() failed with error: " << cudaGetErrorString(result)
                << std::endl;
      return false;
    }

    result = cudaGetDeviceProperties(&properties, device_idx);

    if (result != cudaSuccess) {
      std::cerr << "cudaGetDeviceProperties() failed with error: " << cudaGetErrorString(result)
                << std::endl;
      return false;
    }

    if (properties.major < 8 || (properties.major == 8 && properties.minor < 9)) {
      std::cerr << "CUTLASS's Ada FP8 GEMM example requires a device of compute capability 89 or "
                   "higher.\n"
                << std::endl;
      return false;
    }

    if (properties.sharedMemPerBlockOptin < smem_size) {
      std::cerr << "Insufficient shared memory. Need " << smem_size << ", but device only has "
                << properties.sharedMemPerBlockOptin << std::endl;
      return false;
    }

    return true;
  }

  /// Executes one test
  bool
  run(Options &options) {
    // Waive test if insufficient CUDA device
    if (!sufficient()) {
      std::cerr << "Insufficient resources to run the kernel." << std::endl;
      return false;
    }

    this->initialize(options);

    auto problem_size = options.problem_size;
    int m = problem_size.m(), n = problem_size.n(), k = problem_size.k();

    bool fast_accum = (dt_conf.is_input_fp8() ? _True{} : _False{}) && kUseFastAccum;
    auto v2_meta = make_gemm_v2_meta(fast_accum);
    auto meta =
        make_gemm_meta(dt_conf, get_arch(), _CommNone{}, cute::C<layout>{}, _GemmV2{}(), v2_meta);

    auto rt_conf = make_runtime_config(m, n, k);
    auto hparams = OpRegistry::instance().get_hparams(meta, rt_conf);
    auto gemm_op = OpRegistry::instance().get_op(meta, hparams);

    auto stream = nullptr;
    auto args = GemmFP8Arguments{
        m,
        n,
        k,
        options.alpha,
        options.beta,
        tensor_A.device_data(),       // input
        tensor_B.device_data(),       // weight
        tensor_C.device_data(),       // nullptr
        tensor_Aux.device_data(),     // nullptr
        tensor_D.device_data(),       // output
        tensor_Vector.device_data(),  // nullptr
        abs_max_Aux.device_data(),    // nullptr
        abs_max_D.device_data(),      // nullptr
        scale_A.device_data(),        //
        scale_B.device_data(),
        scale_C.device_data(),     // nullptr
        scale_D.device_data(),     //
        scale_Aux.device_data()};  // nulptr

    int64_t workspace_size = gemm_op->get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    gemm_op->run(args, workspace.get(), stream);

    //
    // Verify
    //

    bool passed = true;
    if (options.reference_check) {
      passed &= this->verify(options);
    } else {
      std::cout << "Skipped reference check" << std::endl;
    }
    if (!passed) {
      exit(-1);
    }

    //
    // Warm up
    //

    for (int i = 0; i < options.warmup_iterations; ++i) {
      gemm_op->run(args, workspace.get(), stream);
    }

    //
    // Profile
    //

    cudaEvent_t events[2];
    cudaError_t error;
    for (auto &event : events) {
      error = cudaEventCreate(&event);
      if (error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(error) << std::endl;
        return false;
      }
    }

    // Record an event at the start of a series of GEMM operations
    error = cudaEventRecord(events[0]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Run profiling loop
    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm_op->run(args, workspace.get(), stream);
    }

    // Record an event when the GEMM operations have been launched.
    error = cudaEventRecord(events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Wait for work on the device to complete.
    error = cudaEventSynchronize(events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(error) << std::endl;
      return false;
    }

    // Compute average runtime and GFLOPs.
    runtime_ms = runtime_ms / float(options.iterations);
    float gflops = options.gflops(runtime_ms / 1000.0f);

    std::cout << "Problem size: " << options.problem_size.m() << 'x' << options.problem_size.n()
              << 'x' << options.problem_size.k() << std::endl;
    std::cout << "Runtime (ms): " << runtime_ms << std::endl;
    std::cout << "GFLOPs/sec:   " << gflops << std::endl;

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    return passed;
  }
};

template <DataTypeEnum dtype_A, DataTypeEnum dtype_C, GemmLayoutEnum gemm_layout>
void
run(Options &options) {
  using DType = decltype(make_gemm_dtype_config(
      cute::C<dtype_A>{}, cute::C<dtype_A>{}, cute::C<dtype_C>{}, _BF16{}));
  TestbedRunner<DType, gemm_layout> runner;
  runner.run(options);
}

int
main(int argc, char const **argv) {
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  if (options.dtype_A != (int)DataTypeEnum::E4M3 && options.dtype_A != (int)DataTypeEnum::E5M2) {
    std::cerr << "dtype_A should one of 4(E4M3) or 5(E5M2)\n";
    return -1;
  }
  if (options.dtype_C != (int)DataTypeEnum::Void && options.dtype_C != (int)DataTypeEnum::BF16) {
    std::cerr << "dtype_C should one of 0(Void) or 2(BF16)\n";
    return -1;
  }
  if (options.gemm_layout != (int)GemmLayoutEnum::RCR) {
    std::cerr << "gemm_layout should one of 1(RCR)\n";
    return -1;
  }

  if (options.dtype_A == (int)DataTypeEnum::E4M3 && options.dtype_C == (int)DataTypeEnum::Void &&
      options.gemm_layout == (int)GemmLayoutEnum::RCR) {
    std::cout << "run with A type E4M3 C type Void D type BF16 GemmLayout RCR\n";
    run<DataTypeEnum::E4M3, DataTypeEnum::Void, GemmLayoutEnum::RCR>(options);
  }

  if (options.dtype_A == (int)DataTypeEnum::E4M3 && options.dtype_C == (int)DataTypeEnum::BF16 &&
      options.gemm_layout == (int)GemmLayoutEnum::RCR) {
    std::cout << "run with A type E4M3 C type BF16 D type BF16 GemmLayout RCR\n";
    run<DataTypeEnum::E4M3, DataTypeEnum::BF16, GemmLayoutEnum::RCR>(options);
  }

  if (options.dtype_A == (int)DataTypeEnum::E5M2 && options.dtype_C == (int)DataTypeEnum::Void &&
      options.gemm_layout == (int)GemmLayoutEnum::RCR) {
    std::cout << "run with A type E5M2 C type Void D type BF16 GemmLayout RCR\n";
    run<DataTypeEnum::E5M2, DataTypeEnum::Void, GemmLayoutEnum::RCR>(options);
  }

  if (options.dtype_A == (int)DataTypeEnum::E5M2 && options.dtype_C == (int)DataTypeEnum::BF16 &&
      options.gemm_layout == (int)GemmLayoutEnum::RCR) {
    std::cout << "run with A type E5M2 C type BF16 D type BF16 GemmLayout RCR\n";
    run<DataTypeEnum::E5M2, DataTypeEnum::BF16, GemmLayoutEnum::RCR>(options);
  }

  return 0;
}
