// clang-format off
#include "flux/op_registry.h"
namespace bytedance::flux {
using namespace cute;

static int config_ag_gemm_kernel_sm80_tp2_nnodes1 = []() {
  auto &inst = TuningConfigRegistry::instance();
  // PCIe
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}()),_Sm80{}(),_AGKernel{}(),_RRR{}(),_GemmV2{}()),make_runtime_config(8192,24576,12288,make_all_gather_runtime_config(2,1,0)),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64l,32l),cute::make_tuple(16l,8l,16l),_StreamkDP{}()),None{},cute::make_tuple(128l,256l,32l),_GemmStreamK{}(),3,_RasterAlongM{}()));
  // NVlink
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}()),_Sm80{}(),_AGKernel{}(),_RRR{}(),_GemmV2{}()),make_runtime_config(8192,24576,12288,make_all_gather_runtime_config(2,1,0)),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64l,32l),cute::make_tuple(16l,8l,16l),_StreamkDP{}()),None{},cute::make_tuple(128l,256l,32l),_GemmStreamK{}(),3,_RasterAlongM{}()));
  return 0;
}();
}
// clang-format on
