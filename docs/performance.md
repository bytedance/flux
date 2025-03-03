## Performance Report

---

### Dense MLP
For dense MLP, we measured performance of kernels on L20 GPUs. Each machine has 8 GPUs, with a TP size set to 8. The table below shows the performance comparison between flux and torch+nccl. It can be observed that by overlapping fine-grained computation and communication, Flux is able to effectively hide a significant portion of the communication time.

#### M=4096, K=12288, N=49152, TP=8

> 💡 Hint: The shape of the first gemm is (MxK)@(KxN) and the shape of the second gemm is (MxN)@(NxK).

```bash
./launch.sh test/python/ag_gemm/test_ag_kernel.py 4096 49152 12288 --dtype=float16 --iters=10
./launch.sh test/python/gemm_rs/test_gemm_rs.py 4096 12288 49152 --dtype=float16 --iters=10
```

|  | Torch GEMM | Torch Comm | Torch Total | Flux GEMM | Flux Comm | Flux Total |
| --- | --- | --- | --- | --- | --- | --- |
| AG+GEMM (L20) | 5.746 ms | 11.200 ms | **16.946 ms** | 5.316 ms | 0.587 ms | **5.903 ms** |
| GEMM+RS (L20) | 5.392 ms | 12.273 ms | **17.664 ms** | 5.389 ms | 0.341 ms | **5.730 ms** |

AG refers to AllGather. Thus AG+GEMM refers to the first layer of an MLP.
RS refers to ReduceScatter. Thus GEMM+RS refers to the second layer of an MLP.

---

### MoE MLP
For MoE MLP, we measure the performance of kernels on L20 and H100 GPUs. Each machine has 8 GPUs. H100 GPUs are connected via NVLink and L20 GPUs are connected via PCIe.
The performance of flux kernels is shown in the tables below.
Specifically, the torch implementation for MoE layer 0 includes: all-gather + scatter + gemm, and the torch implementation for MoE layer 1 includes: gemm + topk-reduce + reduce-scatter. Flux's optimized kernels show better performance.

#### M=8192, K=8192, N=8192, TP=8, EP=1, num_experts=32, topk=4


>💡 The explanation of M: the input token number on each GPU is M/TP = 8192/8 = 1024. After all-gather, each rank has 1024 * world_size = 1024 * 8 = 8192 tokens. With each token picks topk=4 experts, the token length is further expanded to 32768 as the M dimension for the groupgemm. The size of the M dimension of the intermediate tensor between MoE layer0 and layer1 is then also 32768. After MoE layer1, the token length is reduced to 1024.

```bash
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py
```
|  | Torch | Flux |
| --- | --- | --- |
| MoE layer0 (L20) | 26.792ms | 7.813ms |
| MoE layer1 (L20) | 42.092ms | 7.976ms |
| MoE layer0 (H100) | 4.702ms | 1.146ms |
| MoE layer1 (H100) | 10.452ms | 1.863ms |

#### M=8192, K=8192, N=8192, TP=4, EP=2, num_experts=32, topk=4

```bash
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py -E 2
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -E 2 -T 4
```

|  | Torch | Flux |
| --- | --- | --- |
| MoE layer0 (L20) | 26.911ms | 8.727ms |
| MoE layer1 (L20) | 34.850ms | 8.445ms |
| MoE layer0 (H100) | 4.069ms | 1.335ms |
| MoE layer1 (H100) | 6.672ms | 1.685ms |

#### M=8192, K=8192, N=8192, TP=8, EP=1, num_experts=8, topk=2

```bash
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py --G 8 --topk 2
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -G 8 --topk 2 -M 16384
```

|  | Torch | Flux |
| --- | --- | --- |
| MoE layer0 (L20) | 19.734ms | 5.901ms |
| MoE layer1 (L20) | 28.046ms | 6.744ms |
| MoE layer0 (H100) | 2.124ms | 0.657ms |
| MoE layer1 (H100) | 4.561ms | 1.019ms |


#### M=8192, K=8192, N=8192, TP=4, EP=2, num_experts=8, topk=2
```bash
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py -E 2 --G 8 --topk 2
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -E 2 -T 4 -G 8 --topk 2 -M 16384
```

|  | Torch | Flux |
| --- | --- | --- |
| MoE layer0 (L20) | 21.68ms | 6.155ms |
| MoE layer1 (L20) | 25.118ms | 7.009ms |
| MoE layer0 (H100) | 2.284ms | 0.955ms |
| MoE layer1 (H100) | 3.254ms | 0.981ms |

---
For more guide on how to use MoE kernels in Flux, please refer to docs/mlsys_comet_ae.md.
