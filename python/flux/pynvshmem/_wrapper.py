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

import torch
import builtins
from typing import Union
import flux

__all__ = [
    "CUDA_STREAM_TYPE",
    "put_tensor_on_stream",
    "_stream_raw",
]

POINTER_TYPE = builtins.int
CUDA_STREAM_TYPE = Union[torch.cuda.streams.Stream, POINTER_TYPE]


def _stream_raw(stream: CUDA_STREAM_TYPE):
    return stream.cuda_stream if isinstance(stream, torch.cuda.streams.Stream) else stream


def put_tensor_on_stream(
    dst_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    peer: int,
    stream: CUDA_STREAM_TYPE,
):
    assert dst_tensor.device == src_tensor.device
    assert dst_tensor.dtype == src_tensor.dtype
    assert dst_tensor.numel() == src_tensor.numel()
    return flux.pynvshmem_mod._pynvshmemx_putmem_on_stream(
        dst_tensor.data_ptr(),
        src_tensor.data_ptr(),
        dst_tensor.numel() * dst_tensor.element_size(),
        peer,
        _stream_raw(stream),
    )
