# Guide for Comet (MLSys25 Artifact Evaluation)
This git repo (Flux) contains the components for the paper "Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts". The main code of Comet is located in the `src` directory. In detail, the implementation of MoE layer0 is in `src/moe_ag_scatter` and the implementation of MoE layer1 is in `src/moe_gather_rs`.

---

## Measure the MoE layer and E2E model latency
We first provide a guide to measure the latency of MoE layer and E2E model using Comet with Megatron-LM, on a node with 8 GPUs.

### Prepare the environment
```bash
# Under your workspace
# Install some basic dependencies

pip3 install flash-attn --no-build-isolation
pip3 install transformer_engine[pytorch]
git clone https://github.com/NVIDIA/apex
pushd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
pip install git+https://github.com/fanshiqing/grouped_gemm@v1.0
pip install regex

# Megatron-LM
git clone https://github.com/ZSL98/Megatron-LM.git
# fastmoe (baseline1)
git clone https://github.com/ZSL98/fastmoe.git && cd fastmoe && python setup.py install && pip install dm-tree && cd ..
# tutel (baseline2)
git clone https://github.com/ZSL98/tutel && cd tutel && python setup.py install && cd ..

```

### Run the tests

```bash
cd Megatron-LM
bash ./grid_test.sh # Record the single MoE layer results to timing_results.csv
bash ./e2e_grid_test.sh # Record the e2e model results to e2e_timing_results.csv
```
You can modify the parameters such as `NUM_TOKENS` and `EXPERT_NUM` to see the results under different configurations. The scripts' output can be found in `Megatron-LM/timing_results.csv` and `Megatron-LM/e2e_timing_results.csv`. The feasibility of the scripts has been tested on both 8 L20 GPUs and 8 H100 GPUs.

---

## Implement an MoE layer
It is a good starting point to understand how to use the pluggable Flux kernels to build a MoE layer. Readers can refer to the `MoE_layer_flux` class in `Megatron-LM/test/unit_tests/transformer/moe/test_moe.py` for detailed implementation. We give a general and simplified flow for Hopper architecture here.

```python
class MoE_layer_flux(torch.nn.Module):
    def __init__(self,...):
        super().__init__()
        self.ctx = MoeMlpCtx(...)
        self.flux_ag_op = flux.GemmGroupedV3AGScatter(...)
        self.activation = torch.nn.functional.gelu
        self.flux_rs_op = flux.GemmGroupedV3GatherRS(...)

    def forward(self, x):
        # The router is omitted here
        self.flux_ag_op.forward_multiple_weights(
            inputs_shard=self.ctx.inputs_shard,
            weights=self.ctx.weights,
            splits_gpu=self.ctx.splits_gpu,
            scatter_index=self.ctx.scatter_index,
            outputs_buf=self.ctx.outputs,
            ...
        )
        self.ctx.outputs[0] = self.activation_func(self.ctx.outputs[0])
        mlp_output = self.flux_rs_op.forward_gather_rs(
            input=self.ctx.outputs[0],
            weight=self.ctx._weight,
            split_cpu=self.ctx.splits_cpu,
            scatter_idx=self.ctx.scatter_index.view(-1),
            ...
        )
        return mlp_output
```

### Key functions and parameters
The kernels used on Hopper architecture are `GemmGroupedV3AGScatter` and `GemmGroupedV3GatherRS`. The `GemmGroupedV3AGScatter` kernel is used for all-gather + scatter + gemm. The `GemmGroupedV3GatherRS` kernel is used for gemm + topk-reduce + reduce-scatter. For kernels on previous architectures (i.e., sm80 and sm89), the kernels are `GemmGroupedV2AGScatterOp` and `GemmGroupedV2GatherRSOp`.

The `splits_gpu` and `splits_cpu` are used to specify the number of tokens that are distributed across experts. The sum of `splits_gpu` and `splits_cpu` should be equal to `n_tokens*topk`. The `scatter_index` is used to specify the index of the position that each token is assigned to. The `scatter_index` is a 2D tensor with shape [n_tokens, topk].

---
Then the MoE layer can be used as a independent module and be plugged into any frameworks you are using. To understand more on the detailed design of the kernels, please refer to [Design Guide](https://github.com/ZSL98/flux/blob/comet_clean/docs/design.md).