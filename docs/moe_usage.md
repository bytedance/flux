

## Integrate an MoE layer
Flux kernels are pluggable and can be used to implement an MoE layer easily. The following code snippet shows how to implement an MoE layer using the Flux kernels.
Readers can refer to the `MoE_layer_flux` class in https://github.com/ZSL98/Megatron-LM/blob/main/tests/unit_tests/transformer/moe/test_moe.py for detailed implementation. We give a general and simplified flow for Hopper architecture here.

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

Then the MoE layer can be used as a independent module and be plugged into any frameworks you are using. To understand more on the detailed design principles of the kernels, please refer to [Design Guide](https://github.com/bytedance/flux/blob/main/docs/design.md).

## Compare with native pytorch code

An MoE layer can be implemented using native pytorch, composed of several sequential operations, while Flux leverage single independent kernels to achieve so. The following code snippet shows how to implement the layer0 of an MoE layer0 using native pytorch.

```python
def comm_impl(ctx):
    torch.distributed.all_gather_into_tensor(ctx.inputs, ctx.inputs_shard, group=TP_GROUP)
    if ctx.input_scale is not None:
        for i in range(ctx.weight_groups):
            torch.distributed.all_gather_into_tensor(
                ctx.full_input_scale[i], ctx.input_scale[i], group=TP_GROUP
            )

def scatter_impl(ctx):
    ctx.scatter_inputs.copy_(torch.index_select(ctx.inputs, dim=0, index=ctx.gather_index))

def gemm_impl(ctx):
    offset = 0
    expert_id_offset = ctx.nexperts_ep * ctx.ep_rank
    input_offset = torch.sum(ctx.splits_cpu[:expert_id_offset])
    for exp_id in range(ctx.nexperts_ep):
        nxt_offset = offset + ctx.splits_cpu[exp_id + expert_id_offset]
        if nxt_offset - offset > 0:
            exp_input = ctx.scatter_inputs[offset + input_offset : nxt_offset + input_offset]
            for i in range(ctx.weight_groups):
                out = ctx.outputs[i][offset:nxt_offset]
                torch.matmul(
                    exp_input,
                    ctx.weights[i][exp_id].t(),
                    out=out,
                )
                out.mul_(ctx.output_scale[i][exp_id])

        offset = nxt_offset

def torch_moe_ag_gemm(ctx: MoeMlp1Ctx):
    gate_impl(ctx)
    comm_impl(ctx)
    scatter_impl(ctx)
    gemm_impl(ctx)

```

While Flux only needs a single kernel to achieve the same functionality, as below:

```python
def flux_moe_ag_gemm(ctx: MoeMlp1Ctx):    
    tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=DIST_ENV.NNODES, ep_group=EP_GROUP)
    moe_args = flux.MoeArguments(
        max_ntokens=ctx.b * ctx.s,
        hidden=ctx.h,
        ffn_hidden=ctx.ffn_size,
        nexperts=ctx.nexperts,
        topk=ctx.topk,
        input_dtype=ctx.inputs_shard.dtype,
        output_dtype=ctx.outputs[0].dtype,
    )

    op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    op.forward_multiple_weights(
        inputs_shard=ctx.inputs_shard,
        weights=ctx.weights,
        splits_gpu=ctx.splits_gpu,
        scatter_index=ctx.scatter_index,
        output_scale=ctx.output_scale,
        outputs_buf=ctx.outputs,
        fast_accum=ctx.fast_accum,
    )
```

For the second layer of an MoE layer, we also give a comparison between native pytorch and Flux.
The native pytorch code is as below:

```python
def torch_moe_gemm_rs(
    input: torch.Tensor,
    weight: torch.Tensor,
    split_cpu: torch.Tensor,
    token_index: torch.Tensor,
    topk_index: torch.Tensor,
    topk: int,
    input_scale: Union[torch.Tensor, List[torch.Tensor], None],
    weight_scale: Union[torch.Tensor, List[torch.Tensor], None],
    output_vec_scale: Union[torch.Tensor, List[torch.Tensor], None],
):
    output_type = inputs[0].dtype
    full_output = torch.zeros((args.M, args.N), dtype=output_type, device=inputs[0].device)

    acc = 0
    output_list = []

    for exp_id in range(weight.size(0)):
        scale_v = weight_scale[exp_id].item() * input_scale[0].item()
        exp_w = weight[exp_id]
        Mi = split_cpu[exp_id + eid_start]
        exp_input = input[acc : acc + Mi]
        acc += Mi
        output_list.append(scale_v * torch.matmul(exp_input, exp_w.t()))
        
    output = torch.concat(output_list)
    output = (output * output_vec_scale.unsqueeze(1)).to(output_type)
    new_index = (
        topk * token_index[ep_rank_m_start:ep_rank_m_end]
        + topk_index[ep_rank_m_start:ep_rank_m_end]
    )

    output1 = torch.zeros_like(full_output)
    output1[new_index] = output
    full_output += output1
        
    topk_reduce = full_output.view(
        (full_output.size(0) // topk, topk, full_output.size(1))
    ).sum(1)
    output2 = torch.zeros(
        (full_output.size(0) // TP_GROUP.size() // topk, full_output.size(1)),
        dtype=topk_reduce.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed.reduce_scatter_tensor(output2, topk_reduce, group=TP_GROUP)
    return output2
```
FLux's implementation is as simple as:

```python
def flux_moe_gemm_rs(
    input: torch.Tensor,
    weight: torch.Tensor,
    split_cpu: torch.Tensor,
    max_m: int,
    topk: int,
    routing_idx: torch.Tensor,
    input_scale: Union[torch.Tensor, List[torch.Tensor], None],
    weight_scale: Union[torch.Tensor, List[torch.Tensor], None],
    output_vec_scale: Union[torch.Tensor, List[torch.Tensor], None],
):
    n_dim = args.N
    op = flux.GemmGroupedV3GatherRS(
        args.G, max_m, n_dim, topk, RANK, WORLD_SIZE, args.T, args.E, args.inputgroups
    )

    return op.forward_gather_rs(
        input,
        weight,
        split_cpu,
        routing_idx,
        input_scale,
        weight_scale,
        output_vec_scale,
        args.fastacc,
        args.sm_margin,
    )
```