################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
import os
import torch
import numpy as np
import datetime
import torch.distributed
import pynvshmem

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = int(os.environ.get("MASTER_PORT", 10000))

torch.cuda.set_device(LOCAL_RANK)

torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_printoptions(precision=8)
torch.manual_seed(3 + RANK)
torch.cuda.manual_seed_all(3 + RANK)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
np.random.seed(3 + RANK)


torch.distributed.init_process_group(
    backend="nccl",
    world_size=WORLD_SIZE,
    rank=RANK,
    timeout=datetime.timedelta(seconds=1800),
)
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("before init nvshmem with c10d::ProcessGroup")
    pynvshmem.init_with_c10d_pg(TP_GROUP)
    print("after init nvshmem with c10d::ProcessGroup")

    torch.cuda.synchronize()
    t = pynvshmem.create_tensor([1024], torch.int)
    print("create torch tensor with nvshmem")
    torch.cuda.synchronize()
    print(t)
    pynvshmem.int_p(t.data_ptr(), TP_GROUP.rank(), (RANK + 1) % WORLD_SIZE)
    print("after put_rank_to_next")
    print(t.to(torch.int32))
