### FAQ (Frequently Asked Questions)

#### Common questions:

1. **Q:** What kernels do Flux support and how are them named?

    **A:** Flux mainly supports the following kernels:
    - Dense MLP layer0 (AllGather + GEMM) in `src/ag_gemm`
    - Dense MLP layer1 (GEMM + ReduceScatter) in `src/gemm_rs`
    - MoE layer0 (AllGather + Scatter + GroupGEMM) in `src/moe_ag_scatter`
    - MoE layer1 (GroupGEMM + Gather + Topk-reduce + ReduceScatter) in `src/moe_gather_rs`

    Flux supports MoE kernels with tensor parallelism/expert parallelism/tensor+expert parallelism. You can get a minimal example of a MoE layer with EP=4 in `examples/moe_flux_only.py` (Note that sequence parallelism is enabled and the ffn_tp_size is 2). There is also an illustration as `docs/assets/toy_example.png` for this toy example to help you understand the workflow in this TP+EP MoE case better. In this case, the communication of EP is also overlapped by GroupGEMM in Flux's implementation.
    Detailed information about the kernels can be found in the [Design Guide](https://github.com/bytedance/flux/blob/main/docs/design.md).

#### Connection problems

1. **Q:** The NCCL/NVSHMEM connection hangs/fails when initializing Flux.

    **A:** If you encounter a NCCL/NVSHMEM connection problem, that may be the problem of the network configurations inside the `launch.sh` script. A possible solution is to export a proper `NCCL_SOCKET_IFNAME` variable manually. Try to set it to the first name you get from `ifconfig` (e.g., `export NCCL_SOCKET_IFNAME=bond0`).


    If you still cannot establish connection, try setting some more environment variables in the launch script which will describe your network configuration better:

    - `export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0` 
    - `export NVSHMEM_IB_ADDR_FAMILY=AF_INET6`
    - `export NVSHMEM_SYMMETRIC_SIZE=10000000000`

#### Installation problems

1. **Q:** The installation takes too long.

    **A:** Set `export OMP_NUM_THREADS=128` before installation, higher thread num may incur higher compiling speed.

#### Performance problems

1. **Q:** The performance of kernels is not as good as I expected.

    **A:** You may need to tune the kernels because the performance of kernels can vary on different hardwares and with different shapes. About how to tune the kernels, please refer to [tuning guide](https://github.com/bytedance/flux/blob/main/docs/tuning_guide.md).