import os
import torch
import flux
from flux.testing import gen_moe_gating_args, moe_gather_rs_forward_torch, MoeAgScatterWithTorch

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
DIST_ENV = flux.get_dist_env(deterministic=False)
TP_GROUP = DIST_ENV.get_world()
EP_GROUP = None
torch.cuda.set_device(DIST_ENV.LOCAL_RANK)

def init_ep_group(ep_size: int):
    global EP_GROUP
    ffn_tp_size = TP_GROUP.size() // ep_size
    temp_groups, ep_groups = [], []
    for i in range(ffn_tp_size):
        ranks = list(range(i, DIST_ENV.WORLD_SIZE, ffn_tp_size))
        temp_groups.append(ranks)
    for group in temp_groups:
        for i in range(0, len(group), ep_size):
            ep_groups.append(group[i : i + ep_size])
    for ranks in ep_groups:
        group = DIST_ENV.new_group(ranks)
        if DIST_ENV.RANK in ranks:
            EP_GROUP = group

flux.init_flux_shm(TP_GROUP)
# The line below indicates EP=4
init_ep_group(ep_size=4)

class MoeMlp1Ctx():
    naive_impl = True

    h = 4096
    ffn_size = 14336
    nexperts = 8
    topk = 2
    ntokens = 1024
    tp_rank = TP_GROUP.rank()
    tp_size = TP_GROUP.size()
    ep_rank = EP_GROUP.rank()
    ep_size = EP_GROUP.size()

    ffn_tp_size = tp_size // ep_size
    ffn_size_shard = ffn_size // ffn_tp_size
    nexperts_ep = nexperts // ep_size
    ntokens_shard = ntokens // tp_size

    data_type = torch.float16
    device = torch.cuda.current_device()

    # Dummy token routing information
    moe_gating_args = gen_moe_gating_args(nexperts, topk, ntokens)
    splits_gpu = moe_gating_args.splits_gpu
    splits_cpu = moe_gating_args.splits_gpu.to("cpu")
    scatter_index = moe_gating_args.scatter_index
    gather_index = moe_gating_args.gather_index
    topk_index = moe_gating_args.topk_index
    eid_start = nexperts_ep * ep_rank
    ep_rank_m_start = torch.sum(splits_cpu[:eid_start])
    nrows_ep = torch.sum(splits_cpu[nexperts_ep * ep_rank : nexperts_ep * (ep_rank + 1)])

    # Dummy inputs and weights
    inputs_shard = (torch.rand((ntokens_shard, h), dtype=data_type, device=device) * 0.01)
    weight0 = (torch.rand((nexperts_ep, ffn_size_shard, h), dtype=data_type, device=device,) * 0.01)
    weight1 = torch.rand((nexperts_ep, h, ffn_size_shard), dtype=data_type, device=device) - 0.5

    # Buffers
    inputs = (torch.rand((ntokens, h), dtype=data_type, device=device))
    scatter_inputs = torch.zeros((ntokens * topk, h), dtype=data_type, device=device)
    intermediate_output = torch.zeros((nrows_ep, ffn_size_shard), dtype=data_type, device=device)

class MoE_layer_flux(torch.nn.Module):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        tp_env = flux.DistEnvTPWithEP(tp_group=TP_GROUP, nnodes=1, ep_group=EP_GROUP)
        moe_args = flux.MoeArguments(
            max_ntokens=ctx.ntokens,
            hidden=ctx.h,
            ffn_hidden=ctx.ffn_size,
            nexperts=ctx.nexperts,
            topk=ctx.topk,
            input_dtype=ctx.data_type,
            output_dtype=ctx.data_type,
        )
        self.flux_ag_op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
        self.flux_rs_op = flux.GemmGroupedV3GatherRS(ctx.nexperts, ctx.ntokens * ctx.topk, ctx.h, ctx.topk, RANK, WORLD_SIZE, ctx.ffn_tp_size, ctx.ep_size, 1)

    def forward(self):
        # Token routing is omitted
        # MLP layer 0 (dispatch and GEMM0)
        self.flux_ag_op.forward(
            inputs_shard=self.ctx.inputs_shard,
            weights=self.ctx.weight0,
            splits_gpu=self.ctx.splits_gpu,
            scatter_index=self.ctx.scatter_index,
            outputs_buf=self.ctx.intermediate_output,
        )
        # Activation
        self.ctx.intermediate_output = torch.nn.functional.gelu(self.ctx.intermediate_output)
        # MLP layer 1 (GEMM1 and combine)
        mlp_output = self.flux_rs_op.forward_gather_rs(
            input=self.ctx.intermediate_output,
            weight=self.ctx.weight1,
            splits_cpu=self.ctx.splits_cpu,
            routing_idx=self.ctx.scatter_index.view(-1),
        )
        return mlp_output

if __name__ == "__main__":
    moe_ctx = MoeMlp1Ctx()
    flux_moe = MoE_layer_flux(moe_ctx).cuda().to(torch.float16)
    flux_output = flux_moe()
