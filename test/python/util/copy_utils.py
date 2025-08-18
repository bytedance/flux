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
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from cuda import cuda, cudart


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def _get_nvcc_bin():
    # Try finding NVCC
    nvcc_bin: Optional[Path] = None
    if nvcc_bin is None and os.getenv("CUDA_HOME"):
        # Check in CUDA_HOME
        cuda_home = Path(os.getenv("CUDA_HOME"))
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if nvcc_bin is None:
        # Check if nvcc is in path
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin is not None:
            nvcc_bin = Path(nvcc_bin)
    if nvcc_bin is None:
        # Last-ditch guess in /usr/local/cuda
        cuda_home = Path("/usr/local/cuda")
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise FileNotFoundError(f"Could not find NVCC at {nvcc_bin}")
    return str(nvcc_bin)


def load_as_np_bytes(path: str):
    with open(path, "rb") as f:
        ptx = f.read()
    return np.char.array(ptx)


def compile_kernel(source_name: str):
    # don't use nvrtc. hard to include files such as <cuda_fp16.h>. see to how jitify does.
    err, device_id = cudart.cudaGetDevice()
    CUDA_CHECK(err)
    err, prop = cudart.cudaGetDeviceProperties(device_id)
    CUDA_CHECK(err)
    arch_major, arch_minor = prop.major, prop.minor
    CUDA_HOME = os.getenv("CUDA_HOME")
    if CUDA_HOME == None:
        CUDA_HOME = os.getenv("CUDA_PATH")
    if CUDA_HOME == None:
        CUDA_HOME = "/usr/local/cuda"
    include_dirs = os.path.join(CUDA_HOME, "include")
    # Compile program
    opts = [
        "-gencode",
        f"arch=compute_{arch_major}{arch_minor},code=sm_{arch_major}{arch_minor}",
        "-std=c++17",
        "-O2",
    ]

    infile = Path(__file__).parent / "cuda_kernels" / f"{source_name}.cu"
    output_ptx = f"/tmp/{source_name}.ptx"
    output = subprocess.run(
        [_get_nvcc_bin()] + opts + ["-ptx", "-o", output_ptx, "-I", include_dirs, str(infile)],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )

    return load_as_np_bytes(output_ptx)


def load_kernel(ptx: Union[np.array, str], kernel_name: str, device_id: int = None):
    if isinstance(ptx, str) or isinstance(ptx, Path):
        ptx = load_as_np_bytes(ptx)
    if device_id is None:
        device_id = torch.cuda.current_device()
    # Note: Incompatible --gpu-architecture would be detected here
    with torch.cuda.device(device_id):
        (err,) = cudart.cudaFree(0)  # unsure cuda context init
        CUDA_CHECK(err)
        err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
        CUDA_CHECK(err)
        err, kernel = cuda.cuModuleGetFunction(module, bytes(kernel_name, "ascii"))
        CUDA_CHECK(err)
        return kernel


class LazyMultiDeviceKernel:
    def __init__(self, source_name, kernel_name) -> None:
        self.source_name = source_name
        self.kernel_name = kernel_name
        self.kernels = {}
        self.ptx = None

    def kernel(self):
        if self.ptx is None:
            self.ptx = compile_kernel(self.source_name)
        current_device = torch.cuda.current_device()
        if current_device not in self.kernels:
            self.kernels[current_device] = load_kernel(self.ptx, self.kernel_name, current_device)
        return self.kernels[current_device]


copy_kernel = LazyMultiDeviceKernel("copy_kernel", "copy_kernel")
reduce_kernel = LazyMultiDeviceKernel("reduce_kernel", "reduce_kernel")


def copy_cudaMemcpyAsync(
    dst_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    stream: Optional[torch.cuda.Stream] = None,
):
    def _bytes(tensor: torch.Tensor):
        return tensor.numel() * tensor.dtype.itemsize

    assert _bytes(dst_tensor) == _bytes(src_tensor)
    stream = torch.cuda.current_stream() if stream is None else stream
    assert isinstance(stream, torch.cuda.Stream)
    (err,) = cudart.cudaMemcpyAsync(
        dst_tensor.data_ptr(),
        src_tensor.data_ptr(),
        _bytes(dst_tensor),
        cudart.cudaMemcpyKind.cudaMemcpyDefault,
        stream.cuda_stream,
    )
    CUDA_CHECK(err)


def run_kernel(
    kernel,
    dst_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    num_blocks: int = 4,
    num_threads: int = 1024,
    stream: Optional[torch.cuda.Stream] = None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    assert dst_tensor.is_contiguous()
    assert src_tensor.is_contiguous()
    assert dst_tensor.dtype == src_tensor.dtype and dst_tensor.numel() == src_tensor.numel()
    nbytes = dst_tensor.numel() * dst_tensor.element_size()
    args = np.array(
        [
            x.ctypes.data
            for x in [
                np.array(dst_tensor.data_ptr(), dtype=np.uint64),
                np.array(src_tensor.data_ptr(), dtype=np.uint64),
                np.array(nbytes, dtype=np.uint64),
            ]
        ],
        dtype=np.uint64,
    )
    (err,) = cuda.cuLaunchKernel(
        kernel,
        num_blocks,  # grid x dim
        1,  # grid y dim
        1,  # grid z dim
        num_threads,  # block x dim
        1,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream.cuda_stream,  # stream
        args.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )
    CUDA_CHECK(err)


def copy_continous(
    dst_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    num_blocks: int = 4,
    num_threads: int = 1024,
    stream: Optional[torch.cuda.Stream] = None,
):
    run_kernel(copy_kernel.kernel(), dst_tensor, src_tensor, num_blocks, num_threads, stream)


def reduce_continous(
    dst_tensor: torch.Tensor,
    src_tensor: torch.Tensor,
    num_blocks: int = 4,
    num_threads: int = 1024,
    stream: Optional[torch.cuda.Stream] = None,
):
    run_kernel(reduce_kernel.kernel(), dst_tensor, src_tensor, num_blocks, num_threads, stream)


def copy_tensor_impl(
    from_tensor: torch.Tensor,
    to_tensor: torch.Tensor,
    use_cudaMemcpy,
    num_blocks: int = 4,
    num_threads: int = 1024,
):
    if use_cudaMemcpy:
        to_tensor.copy_(from_tensor)
    else:
        copy_continous(to_tensor, from_tensor, num_blocks, num_threads, torch.cuda.current_stream())


def set_cuda_p2p_access():
    ndevices = torch.cuda.device_count()
    for n in range(ndevices):
        torch.cuda.set_device(n)
        for m in range(ndevices):
            if n != m:
                cudart.cudaDeviceEnablePeerAccess(m, 0)
