// clang-format off
#include "flux/op_registry.h"
namespace bytedance::flux {
using namespace cute;

static int config_ag_gemm_kernel_sm90_tp1_nnodes1 = []() {
  auto &inst = TuningConfigRegistry::instance();
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}()),_Sm90{}(),_AGKernel{}(),_RRR{}(),_GemmV3{}()),make_runtime_config(8192,49152,12288,make_all_gather_runtime_config(1,1,0)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(256l,128l,64l),_GemmDefault{}(),4,_RasterAlongN{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_FP16{}(),_FP16{}(),_Void{}(),_FP16{}()),_Sm90{}(),_AGKernel{}(),_RRR{}(),_GemmV3{}()),make_runtime_config(8192,49152,12288,make_all_gather_runtime_config(1,1,0)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),3,_RasterAlongM{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}()),_Sm90{}(),_AGKernel{}(),_RCR{}(),_GemmV3{}()),make_runtime_config(8192,49152,12288,make_all_gather_runtime_config(1,1,0)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),4,_RasterAlongM{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_FP16{}(),_FP16{}(),_Void{}(),_FP16{}()),_Sm90{}(),_AGKernel{}(),_RCR{}(),_GemmV3{}()),make_runtime_config(8192,49152,12288,make_all_gather_runtime_config(1,1,0)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),4,_RasterAlongM{}()));
  return 0;
}();
}
// clang-format on
