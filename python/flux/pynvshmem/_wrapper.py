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
