# Tuning guide

You may need to tune a kernel before it achieving the best performance. In this guide, we use the tuning process of an **MoE layer0**'s kernel as an example to explain the workflow. Tuning other kernels are similar and more detailed tutorials are on the way.

---

### MoE layer 0 (AllGather + Scatter + GroupGEMM)
To enable tuning for the test demo, you only need to set the `--tune` flag.

```bash
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py --tune
```

Then you would find the profiling result for the top-5 kernels as well as the best kernel configuration printed out. The optimal one is registered automatically.

```c++
====== Profiling Results =======
GemmMeta(dtype=GemmDTypeConfig(a=BF16,b=BF16,c=BF16,d=BF16,acc=FP32,blockscale=FP32),arch=Sm90,comm_op=AGScatter,gemm_layout=RCR,impl=GemmGroupedV3,impl_spec=GemmV3Meta(fast_accum=0,block_scale=0),comm_spec=None)
RuntimeConfig(m=1024,n=1024,k=8192,comm_spec=None)
 * TopK=1 (1.15 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=None,tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterAlongM)
 * TopK=2 (1.15 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(2,1,1),kernel_schedule=Cooperative),comm_spec=None,tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterAlongN)
 * TopK=3 (1.16 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(2,1,1),kernel_schedule=Cooperative),comm_spec=None,tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterAlongM)
 * TopK=4 (1.17 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=None,tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterAlongN)
 * TopK=5 (1.33 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=PingPong),comm_spec=None,tile_shape=(64,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterAlongM)
====== Generated Config Code =======
// clang-format off
#include "flux/op_registry.h"
namespace bytedance::flux {
using namespace cute;

static int config_ag_scatter_sm90 = []() {
  auto &inst = TuningConfigRegistry::instance();
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_BF16{}(),_BF16{}(),_FP32{}(),_FP32{}()),_Sm90{}(),_AGScatter{}(),_RCR{}(),_GemmGroupedV3{}(),make_gemm_v3_meta(false,false),None{}),make_runtime_config(1024,1024,8192,None{}),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,1l,1l),_Cooperative{}()),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterAlongM{}()));
  return 0;
}();
}
// clang-format on
```

The search space for tuning is defined in `src/generator`. For the MoE layer0's kernel, the search space is defined in `src/generator/gen_moe_ag_scatter.cc`. For example, the search space for GEMM tile size is defined as `cute::make_tuple(Shape<Auto, _256, Auto>{}, Shape<Auto, _128, Auto>{})` in #L88. Modify these codes and compile Flux again if you want enlarge the search space.

### MoE layer 1 (GroupGEMM + Gather + Topk-reduce + ReduceScatter)

Tune the MoE layer1 kernel as follows:
```bash
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py --tune
```
Then the profiling result is as follows:

```c++
====== Profiling Results =======
GemmMeta(dtype=GemmDTypeConfig(a=BF16,b=BF16,c=BF16,d=BF16,acc=FP32,blockscale=FP32),arch=Sm90,comm_op=GatherRS,gemm_layout=RCC,impl=GemmGroupedV3,impl_spec=GemmV3Meta(fast_accum=0,block_scale=0),comm_spec=GatherRSMeta(topk=4))
RuntimeConfig(m=8192,n=1024,k=1024,comm_spec=None)
 * TopK=1 (1.87 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=26,n_dim=8192),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=2 (1.88 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=28,n_dim=8192),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)
 * TopK=3 (1.91 ms): GemmHParams(impl_spec=GemmV3HParams(cluster_shape=(1,1,1),kernel_schedule=Cooperative),comm_spec=GatherRSHParams(gather_rs_ctas=30,n_dim=8192),tile_shape=(128,256,64),gemm_kind=GemmDefault,mainloop_stage=0,raster_order=RasterHeuristic)

====== Generated Config Code =======
// clang-format off
#include "flux/op_registry.h"
namespace bytedance::flux {
using namespace cute;

static int config_gather_rs_sm90 = []() {
  auto &inst = TuningConfigRegistry::instance();
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_BF16{}(),_BF16{}(),_FP32{}(),_FP32{}()),_Sm90{}(),_GatherRS{}(),_RCC{}(),_GemmGroupedV3{}(),make_gemm_v3_meta(false,false),make_gather_rs_meta(4)),make_runtime_config(8192,1024,1024,None{}),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,1l,1l),_Cooperative{}()),make_gather_rs_hparams(26,8192),cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  return 0;
}();
}
// clang-format on
```

You can find the configuration space defined in `src/generator/gen_moe_gather_rs.cc`. You may notice that there are only three kernels been profiled in the case above. This is because there are only three qualified kernels in the search space for the configuration in the test demo, as defined in #L90-92 of `src/generator/gen_moe_gather_rs.cc`. The first value in `make_gather_rs_hparams` refers to the number of thread blocks specialized for communication and the second value refers to the size of the hidden dimension. You must make sure at least one hparams is registered here for the shape of the MoE layer1 you want.