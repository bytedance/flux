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
import sys
from contextlib import nullcontext, contextmanager


def _get_torch_fp8_dtypes():
    _torch_fp8_dtypes = []
    try:
        # from v2.1.0
        _torch_fp8_dtypes.append(torch.float8_e4m3fn)
        _torch_fp8_dtypes.append(torch.float8_e5m2)
        # from v2.2.0
        _torch_fp8_dtypes.append(torch.float8_e4m3fnuz)
        _torch_fp8_dtypes.append(torch.float8_e5m2fnuz)
    except Exception:
        pass
    return _torch_fp8_dtypes


def get_arch():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    major = properties.major
    minor = properties.minor
    return major * 10 + minor


def estimate_gemm_sol_time_ms(M: int, N: int, K: int, dtype=torch.bfloat16):
    """refer to this: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/"""

    def _get_flops():
        device_name = torch.cuda.get_device_name(
            torch.cuda.current_device()
        )  # arch is not a good idea. using device name is better.
        is_fp16 = dtype in [torch.bfloat16, torch.float16]
        is_fp8 = is_fp8_dtype(dtype)
        assert is_fp16 or is_fp8
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
        # "NVIDIA A100 80GB PCIe" or "NVIDIA A100-SXM4-80GB" or "A100-SXM4-40GB"
        if device_name.find("A100") >= 0 or device_name.find("A800") >= 0:
            assert is_fp16
            return 312
        # https://resources.nvidia.com/en-us-gpu-resources/a10-datasheet-nvidia?lx=CPwSfP&ncid=no-ncid
        if device_name == "NVIDIA A10":  # No doc from NVIDIA
            return 125 if is_fp16 else 250
        if device_name == "NVIDIA A30":  # No doc from NVIDIA
            return 165 if is_fp16 else 330
        if device_name == "NVIDIA L20":  # No doc from NVIDIA
            return 119 if is_fp16 else 239
        # https://www.nvidia.com/en-us/data-center/l4/
        if device_name == "NVIDIA L4":
            return 121 if is_fp16 else 242
        # https://images.nvidia.com/content/Solutions/data-center/vgpu-L40-datasheet.pdf
        if device_name == "NVIDIA L40":
            return 181 if is_fp16 else 362
        # https://www.nvidia.com/en-us/data-center/l40s/
        if device_name == "NVIDIA L40S":
            return 366 if is_fp16 else 733
        # https://www.nvidia.com/en-us/data-center/h100/
        if device_name.find("H100") >= 0 or device_name.find("H800") >= 0:
            return 989 if is_fp16 else 1979
        if device_name.find("H200") >= 0:
            return 1979 if is_fp16 else 3958
        raise Exception(f"not supported device {device_name}")

    flops = M * N * K * 2
    return flops / _get_flops() / 1e9


def torch_allclose(x, y, rtol, atol):
    if not torch.allclose(x, y, rtol=rtol, atol=atol):
        print(f"shape of x: {x.shape}")
        print(f"shape of y: {y.shape}")

        print("x:", file=sys.stderr)
        print(x, file=sys.stderr)
        print("y:", file=sys.stderr)
        print(y, file=sys.stderr)
        print("x-y", x - y, file=sys.stderr)
        diff_loc = torch.isclose(x, y, rtol=rtol, atol=atol) == False
        print("x diff:", file=sys.stderr)
        print(x[diff_loc], file=sys.stderr)
        print("y diff:", file=sys.stderr)
        print(y[diff_loc], file=sys.stderr)
        num_diff = torch.sum(diff_loc)
        diff_rate = num_diff / (y.shape[0] * y.shape[1])
        print(f"diff count: {num_diff} ({diff_rate*100:.3f}%), {list(y.shape)}", file=sys.stderr)
        max_diff = torch.max(torch.abs(x - y))
        rtol_abs = rtol * torch.min(torch.abs(y))
        print(f"diff max: {max_diff}, atol: {atol}, rtol_abs: {rtol_abs}", file=sys.stderr)
        diff_indices = (diff_loc == True).nonzero(as_tuple=False)
        print(f"diff locations:\n{diff_indices}", file=sys.stderr)
        print("--------------------------------------------------------------\n", file=sys.stderr)
        raise RuntimeError


def get_torch_prof_ctx(do_prof: bool):
    ctx = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )
        if do_prof
        else nullcontext()
    )
    return ctx


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in _get_torch_fp8_dtypes()


@contextmanager
def with_torch_deterministic(mode: bool, warn_only: bool = True):
    old_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    yield
    torch.use_deterministic_algorithms(old_mode, warn_only=warn_only)


__all__ = [
    "is_fp8_dtype",
    "get_arch",
    "estimate_gemm_sol_time_ms",
    "torch_allclose",
    "get_torch_prof_ctx",
    "is_fp8_dtype",
    "with_torch_deterministic",
]
