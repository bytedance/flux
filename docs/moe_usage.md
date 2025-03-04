

## Implement an MoE layer
Flux kernels are pluggable and can be used to implement an MoE layer easily. The following code snippet shows how to implement an MoE layer using the Flux kernels.
Readers can refer to the `MoE_layer_flux` class in `Megatron-LM/test/unit_tests/transformer/moe/test_moe.py` for detailed implementation. We give a general and simplified flow for Hopper architecture here.

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
Then the MoE layer can be used as a independent module and be plugged into any frameworks you are using. To understand more on the detailed design principles of the kernels, please refer to [Design Guide](https://github.com/ZSL98/flux/blob/comet_clean/docs/design.md).