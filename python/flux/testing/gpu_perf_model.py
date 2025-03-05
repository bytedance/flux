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

# TODO(houqi.1993) some code comes from triton. copy but how to write the copyright?
from typing import Optional
import torch
import subprocess
import sys
import functools
import logging

_SHARED_MEMORY_SIZE = {"NVIDIA L20": 101376}


def get_device_shared_memory_size():
    try:
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        return _SHARED_MEMORY_SIZE[name]
    except Exception:
        from cuda import cudart

        code, shared_memory_size = cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0
        )
        assert code == cudart.cudaError_t.cudaSuccess
        return shared_memory_size


def is_fp8_dtype(dtype: torch.dtype):
    return dtype.itemsize == 1 and dtype.is_floating_point


def nvsmi(attrs):
    attrs = ",".join(attrs)
    cmd = ["nvidia-smi", "-i", "0", "--query-gpu=" + attrs, "--format=csv,noheader,nounits"]
    out = subprocess.check_output(cmd)
    ret = out.decode(sys.stdout.encoding).split(",")
    ret = [int(x) for x in ret]
    return ret


@functools.lru_cache()
def get_clock_rate_in_khz():
    try:
        return nvsmi(["clocks.max.sm"])[0] * 1e3
    except FileNotFoundError:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM) * 1e3


@functools.lru_cache()
def get_device_property(device):
    return torch.cuda.get_device_properties(device=device)


@functools.lru_cache()
def get_device_multi_processor_count(device):
    return get_device_property(device).multi_processor_count


def get_max_tensorcore_tflops(dtype, clock_rate, device=None):
    device = device or torch.cuda.current_device()
    num_subcores = get_device_multi_processor_count(device) * 4  # on recent GPUs
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        assert dtype == torch.float16
        ops_per_sub_core = 256  # 2 4x4x4 Tensor Cores
    else:
        if dtype in [torch.float32, torch.int32]:
            ops_per_sub_core = 256
        elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
            ops_per_sub_core = 512  # 16x8x32
        elif is_fp8_dtype(dtype) or dtype == torch.int8:
            ops_per_sub_core = 1024  #
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    logging.info(
        f"num_subcores: {num_subcores} clock_rate: {clock_rate} ops_per_sub_core: {ops_per_sub_core}"
    )
    return tflops


def get_max_simd_tflops(dtype, clock_rate, device=None):
    if not device:
        device = torch.cuda.current_device()

    num_subcores = get_device_multi_processor_count(device) * 4
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8:
        if dtype == torch.float32:
            ops_per_sub_core = 32  # 2*16
        elif dtype == torch.float16:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    else:
        if dtype == torch.float32:
            ops_per_sub_core = 32
        elif dtype in [torch.float16, torch.bfloat16]:
            ops_per_sub_core = 64
        else:
            raise RuntimeError("dtype not supported")
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-9
    return tflops


def get_tensorcore_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = get_device_multi_processor_count(device) * 4  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps)
        / num_subcores
        * get_max_tensorcore_tflops(dtype, get_clock_rate_in_khz(), device)
    )
    return tflops


def get_simd_tflops(device, num_ctas, num_warps, dtype):
    """return compute throughput in TOPS"""
    total_warps = num_ctas * min(num_warps, 4)
    num_subcores = get_device_multi_processor_count(device) * 4  # on recent GPUs
    tflops = (
        min(num_subcores, total_warps)
        / num_subcores
        * get_max_simd_tflops(dtype, get_clock_rate_in_khz(), device)
    )
    return tflops


def get_tflops_approx(device: torch.dtype, num_ctas: int, num_warps: int, dtype: torch.dtype):
    """You may not achieve"""
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8 and dtype == torch.float32:
        return get_simd_tflops(device, num_ctas, num_warps, dtype)
    return get_tensorcore_tflops(device, num_ctas, num_warps, dtype)


def get_full_tflops_approx(dtype: torch.dtype, device: Optional[torch.device] = None):
    prop = torch.cuda.get_device_properties(device)
    return get_tflops_approx(device, prop.multi_processor_count, 4, dtype)


def get_tflops(dtype):
    """TFLOPS with no sparse."""
    device_name = torch.cuda.get_device_name(
        torch.cuda.current_device()
    )  # arch is not a good idea. using device name is better.
    is_fp16 = dtype in [torch.bfloat16, torch.float16]
    is_fp8 = is_fp8_dtype(dtype)
    is_s8 = dtype == torch.int8
    assert is_fp16 or is_fp8 or is_s8
    # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
    # "NVIDIA A100 80GB PCIe" or "NVIDIA A100-SXM4-80GB" or "A100-SXM4-40GB"
    if device_name.find("A100") >= 0 or device_name.find("A800") >= 0:
        assert is_fp16 or is_s8
        return 312 if is_fp16 else 624
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
    if device_name == "NVIDIA H100" or device_name == "NVIDIA H800":
        return 989 if is_fp16 else 1979
    if device_name == "NVIDIA H20":
        return 148 if is_fp16 else 296

    logging.warning(f"device {device_name} not listed in flux. calculate tflops by estimation")

    return get_full_tflops_approx(dtype=dtype)


def get_dram_gbps_by_device_name(device_name: str):
    _DRAM_GBPS = {
        "NVIDIA L20": 864,
        "NVIDIA L4": 300,
        "NVIDIA L40": 864,
        "NVIDIA L40S": 864,
        "NVIDIA H20": 4000,
        "NVIDIA A100 80GB PCIe": 1935,
        "NVIDIA A100-SXM4-80GB": 2039,
        "NVIDIA A100-SXM4-40GB": 1555,
        "NVIDIA H100 SXM": 3958,
        "NVIDIA H100 NVL": 3341,
    }
    return _DRAM_GBPS[device_name]


def get_dram_gbps(device=None):
    try:
        import triton

        return triton.testing.get_dram_gbps(device)
    except:
        return get_dram_gbps_by_device_name(torch.cuda.get_device_name(device))


def estimate_gemm_sol_time_ms(M: int, N: int, K: int, dtype=torch.bfloat16):
    """refer to this: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/"""

    flops = M * N * K * 2
    return flops / get_tflops(dtype=dtype) / 1e9


def _get_output_dtype(input_dtype):
    is_s8 = input_dtype == torch.int8
    is_fp8 = is_fp8_dtype(input_dtype)
    return torch.bfloat16 if is_s8 and is_fp8 else input_dtype


def print_gemm_sol_time(M, N, K, input_dtype, output_dtype=None):
    """return time in milliseconds"""
    output_dtype = output_dtype or _get_output_dtype(input_dtype)
    flops = M * K * N * 2
    bytes_r = (M * K + N * K) * input_dtype.itemsize
    bytes_w = (M * N) * output_dtype.itemsize
    t_c = flops / 1e9 / get_tflops(input_dtype)
    t_r, t_w = bytes_r / 1e6 / get_dram_gbps(), bytes_w / 1e6 / get_dram_gbps()
    print(f"SOL gemm TensorCore {t_c:0.3f}ms Memory read: {t_r:0.3f}ms write: {t_w:0.3f}ms")
    return t_c, t_r, t_w


def print_grouped_gemm_sol_time_ms(
    M: int,
    N: int,
    K: int,
    E: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype = None,
    num_groups: int = 1,
):
    output_dtype = output_dtype or _get_output_dtype(input_dtype)
    flops = M * K * N * 2 * num_groups
    bytes_r = (M * K + N * K * E) * input_dtype.itemsize * num_groups
    bytes_w = (M * N) * output_dtype.itemsize * num_groups
    t_c = flops / 1e9 / get_tflops(input_dtype)
    t_r, t_w = bytes_r / 1e6 / get_dram_gbps(), bytes_w / 1e6 / get_dram_gbps()
    print(f"SOL grouped gemm TensorCore {t_c:0.3f}ms Memory read: {t_r:0.3f}ms write: {t_w:0.3f}ms")
    return t_c, t_r, t_w


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int8,
    ]:
        tflops = get_tflops(dtype)
        tflops_approx = get_full_tflops_approx(dtype)
        print(f"{dtype}: {tflops} TFOLPS or {tflops_approx:0.1f} TFLOPS")
