## Performance Report

### Dense MLP
We measured the examples from the above demo on both A800s and H800s. Each machine has 8 GPUs, with a TP size set to 8. The table below shows the performance comparison between flux and torch+nccl. It can be observed that by overlapping fine-grained computation and communication, Flux is able to effectively hide a significant portion of the communication time

|  | M | K | N | Torch Gemm | Torch NCCL | Torch Total | Flux Gemm | Flux Comm | Flux Total |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|
| AG+Gemm(A800) | 4096 | 12288 | 49152 | 2.438ms | 0.662ms | 3.099ms | 2.378ms | 0.091ms | 2.469ms |
| Gemm+RS(A800) | 4096 | 49152 | 12288 | 2.453ms | 0.646ms | 3.100ms | 2.429ms | 0.080ms | 2.508ms |
| AG+Gemm(H800) | 4096 | 12288 | 49152 | 0.846ms | 0.583ms | 1.429ms | 0.814ms | 0.143ms | 0.957ms |
| Gemm+RS(H800) | 4096 | 49152 | 12288 | 0.818ms | 0.590ms | 1.408ms | 0.822ms | 0.111ms | 0.932ms |

AG refers to AllGather.
RS refers to ReduceScatter.

### MoE MLP
For MoE MLP, we measure the performance of kernels on L20 and H100 GPUs. Each machine has 8 GPUs. 
The performance of flux kernels is shown in the table below.
Specifically, the torch implementation for MoE layer 0 includes: all-gather + scatter + gemm, and the torch implementation for MoE layer 1 includes: gemm + topk-reduce + reduce-scatter. Flux's optimized kernels show better performance.

M=8192, K=8192, N=8192, TP=8, EP=1, num_experts=32, topk=4 (The case in Run Demo)

<aside>
💡
The explanation of M: the input token number on each GPU is M/TP=8192/8=1024. After all-gather, each rank has 1024**world_size=1024**8=8192 tokens. With each token picks topk=4 experts, the token length is further expanded to 32768 as the M dimension for the groupgemm. The size of the M dimension of the intermediate tensor between MoE layer0 and layer1 is then also 32768. After MoE layer1, the token length is reduced to 1024.

</aside>

```
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py
```
|  | Torch | Flux |
| --- | --- | --- |
| MoE layer0 (L20) |  |  |
| MoE layer1 (L20) |  |  |
| MoE layer0 (H100) | 4.702ms | 1.146ms |
| MoE layer1 (H100) | 10.452ms | 1.863ms |

M=8192, K=8192, N=8192, TP=4, EP=2, num_experts=32, topk=4

```
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py -E 2
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -E 2 -T 4
```

|  | Torch | Flux |
| --- | --- | --- |
| MoE layer0 (L20) |  |  |
| MoE layer1 (L20) |  |  |
| MoE layer0 (H100) | 4.069ms | 1.335ms |
| MoE layer1 (H100) | 6.672ms | 1.685ms |

M=8192, K=8192, N=8192, TP=8, EP=1, num_experts=8, topk=2

```
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py --G 8 --topk 2
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -G 8 --topk 2 -M 16384
```

|  | Torch | Flux |
| --- | --- | --- |
| MoE layer0 (L20) |  |  |
| MoE layer1 (L20) |  |  |
| MoE layer0 (H100) | 2.124ms | 0.657ms |
| MoE layer1 (H100) | 4.561ms | 1.019ms |


M=8192, K=8192, N=8192, TP=4, EP=2, num_experts=8, topk=2
```
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py -E 2 --G 8 --topk 2
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py -E 2 -T 4 -G 8 --topk 2 -M 16384
```

|  | Torch | Flux |
| --- | --- | --- |
| MoE layer0 (L20) |  |  |
| MoE layer1 (L20) |  |  |
| MoE layer0 (H100) | 2.284ms | 0.955ms |
| MoE layer1 (H100) | 3.254ms | 0.981ms |