// clang-format off
#include "flux/op_registry.h"
namespace bytedance::flux {
using namespace cute;

static int config_gemm_rs_sm80_tp4_nnodes1 = []() {
  auto &inst = TuningConfigRegistry::instance();
  /// PCIE
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}()),_Sm80{}(),_ReduceScatter{}(),_RRR{}(),_GemmV2{}(),None{},make_reduce_scatter_meta(false,_IntraNodePcie{}())),make_runtime_config(8192,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64l,32l),cute::make_tuple(16l,8l,16l)),None{},cute::make_tuple(128l,256l,32l),_GemmStreamK{}(),3,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_FP16{}(),_FP16{}(),_Void{}(),_FP16{}()),_Sm80{}(),_ReduceScatter{}(),_RRR{}(),_GemmV2{}(),None{},make_reduce_scatter_meta(false,_IntraNodePcie{}())),make_runtime_config(8192,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64l,32l),cute::make_tuple(16l,8l,16l)),None{},cute::make_tuple(128l,256l,32l),_GemmStreamK{}(),4,_RasterHeuristic{}()));
  /// NVLink
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}()),_Sm80{}(),_ReduceScatter{}(),_RCR{}(),_GemmV2{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64l,32l),cute::make_tuple(16l,8l,16l)),None{},cute::make_tuple(128l,256l,32l),_GemmStreamK{}(),3,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}()),_Sm80{}(),_ReduceScatter{}(),_RRR{}(),_GemmV2{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64l,32l),cute::make_tuple(16l,8l,16l)),None{},cute::make_tuple(128l,256l,32l),_GemmStreamK{}(),3,_RasterHeuristic{}()));
  return 0;
}();
}
// clang-format on
