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
import flux
from flux.testing import DTYPE_MAP, init_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("--num_streams", type=int, default=1)
    parser.add_argument(
        "--input_dtype",
        default="float8_e4m3fn",
        type=str,
        choices=["float8_e4m3fn", "float8_e5m2"],
    )
    parser.add_argument(
        "--output_dtype",
        default="",
        type=str,
        choices=["float8_e4m3fn", "float8_e5m2", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--output_transpose", default=False, action="store_true", help="transpose output"
    )

    return parser.parse_args()


if __name__ == "__main__":
    init_seed()
    args = parse_args()
    input_dtype = DTYPE_MAP[args.input_dtype]
    output_dtype = DTYPE_MAP[args.output_dtype]

    flux_op = flux.Quantization(
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        num_streams=args.num_streams,
    )

    input_tensor = torch.rand([args.M, args.N], dtype=input_dtype).cuda()
    flux_out = flux_op.quantize_square_blockwise(input_tensor, args.output_transpose)
    ## TODO: add reference implementation
    flux.bitwise_check(flux_out[0], flux_out[0])
