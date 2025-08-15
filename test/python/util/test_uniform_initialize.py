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

import torch
import torch.distributed

import flux
from flux.testing import DTYPE_MAP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-N", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]
    tensor = torch.zeros(args.M, args.N).cuda().to(dtype)
    flux.uniform_initialize(tensor, 2024, 0.0, 1.0)
    print(tensor)
