################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import argparse
from ast import arg
import datetime
import os
import random
import time
from flux.testing.moe_utils import calc_gather_index
import flux.testing
import torch
import torch.distributed as dist
import flux
from flux.testing import (
    DTYPE_MAP,
    initialize_distributed,
)
import numpy as np
from flux.testing.moe_utils import calc_scatter_index_stable

EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
os.environ["NCCL_DEBUG"] = "ERROR"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
np.random.seed(3)  # need the same seed for all ranks to generate the same token distribution
random.seed(3)
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-N", type=int, default=6144)
    parser.add_argument("--profile", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()
    WORLD_SIZE = EP_GROUP.size()
    op = flux.AsyncSendRecv(args.M, args.N, RANK, EP_GROUP.size(), torch.bfloat16, 2)
    src_buffer_id = 0
    tgt_buffer_id = 1
    # get nvshmem comm buffer
    comm_buffer = op.get_comm_buffer(src_buffer_id)
    # write data into the comm buffer
    comm_buffer.zero_()
    comm_buffer += RANK
    print("input tensor", comm_buffer)
    # send the data to the near peer
    next_peer = (RANK + 1) % WORLD_SIZE
    last_peer = (RANK - 1 + WORLD_SIZE) % WORLD_SIZE
    print(f"next peer {next_peer} last_peer {last_peer}")
    op.write_comm_buffer(next_peer, src_buffer_id, tgt_buffer_id)
    # set the signal
    op.set_signal(next_peer, tgt_buffer_id, RANK)

    # wait the signal
    op.wait_signal_eq(tgt_buffer_id, last_peer)
    # reset the signal to zero
    op.reset_signal(tgt_buffer_id)
    recv_data = op.get_comm_buffer(tgt_buffer_id)
    print(recv_data)
