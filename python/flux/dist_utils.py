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

import os
import datetime
import numpy as np
import torch
import torch.distributed
from typing import Callable


class DistEnv:
    def __init__(self, deterministic: bool = True) -> None:
        self.RANK = int(os.environ.get("RANK", 0))
        self.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        self.LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
        self.NNODES = self.WORLD_SIZE // self.LOCAL_WORLD_SIZE

        self.init_global_group()
        if deterministic:
            self.setup_deterministic(self.RANK)

    def setup_deterministic(self, init_seed: int) -> None:
        os.environ["NCCL_DEBUG"] = "ERROR"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_printoptions(precision=8)
        seed = init_seed + self.RANK
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        np.random.seed(seed)

    def init_global_group(self) -> None:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=self.WORLD_SIZE,
            rank=self.RANK,
            timeout=datetime.timedelta(seconds=1800),
        )

    def get_world(self):
        return torch.distributed.group.WORLD

    def new_group(self, ranks):
        return torch.distributed.new_group(ranks=ranks, backend="nccl")


DIST_ENV = None


def get_dist_env(deterministic: bool = True):
    global DIST_ENV

    if DIST_ENV == None:
        DIST_ENV = DistEnv(deterministic=deterministic)
    return DIST_ENV


def exec_in_rank_order(group: torch.distributed.ProcessGroup, func: Callable):
    for i in range(group.size()):
        if i == group.rank():
            func()
        group.barrier()
        torch.cuda.synchronize()


__all__ = ["get_dist_env", "exec_in_rank_order"]
