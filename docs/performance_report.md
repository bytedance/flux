## Performance
We measured the examples from the above demo on both A800s and H800s. Each machine has 8 GPUs, with a TP size set to 8. The table below shows the performance comparison between flux and torch+nccl. It can be observed that by overlapping fine-grained computation and communication, Flux is able to effectively hide a significant portion of the communication time

|  | M | K | N | Torch Gemm | Torch NCCL | Torch Total | Flux Gemm | Flux Comm | Flux Total |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|
| AG+Gemm(A800) | 4096 | 12288 | 49152 | 2.438ms | 0.662ms | 3.099ms | 2.378ms | 0.091ms | 2.469ms |
| Gemm+RS(A800) | 4096 | 49152 | 12288 | 2.453ms | 0.646ms | 3.100ms | 2.429ms | 0.080ms | 2.508ms |
| AG+Gemm(H800) | 4096 | 12288 | 49152 | 0.846ms | 0.583ms | 1.429ms | 0.814ms | 0.143ms | 0.957ms |
| Gemm+RS(H800) | 4096 | 49152 | 12288 | 0.818ms | 0.590ms | 1.408ms | 0.822ms | 0.111ms | 0.932ms |

AG refers to AllGather.
RS refers to ReduceScatter.