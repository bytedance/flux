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

import ctypes
import importlib
import logging
from pathlib import Path

import torch

FLUX_TORCH_EXTENSION_NAME = "flux_ths_pybind"


def _preload_libs(libname):
    libpath = Path(__file__).parent / "lib" / libname
    try:
        ctypes.CDLL(libpath)
    except OSError as e:
        # Try to load from the LD_LIBRARY_PATH
        logging.debug(f"failed to load {libpath}:\n {e}")
        ctypes.CDLL(libname)


def _load_deps():
    try:
        _preload_libs("libnvshmem_host.so.3")
        _preload_libs("nvshmem_bootstrap_uid.so")
        _preload_libs("nvshmem_transport_ibrc.so.3")
    except Exception as e:
        logging.warning("Failed to load NVSHMEM libs")
    _preload_libs("libflux_cuda.so")
    _preload_libs("libflux_cuda_ths_op.so")


_load_deps()
flux_mod = importlib.import_module(FLUX_TORCH_EXTENSION_NAME)


class NotCompiled:
    pass


def _get_flux_member(member):
    return getattr(flux_mod, member, NotCompiled())


def _get_from_torch_classes(name: str):
    try:
        return getattr(torch.classes.flux, name)
    except:
        return NotCompiled()


bsr_reduce = _get_flux_member("bsr_reduce")
init_flux_shm = flux_mod.init_flux_shm
bitwise_check = flux_mod.bitwise_check
uniform_initialize = flux_mod.uniform_initialize
load_tuning_record = flux_mod.load_tuning_record
create_tensor_list = flux_mod.flux_create_tensor_list
GroupBarrier = flux_mod.GroupBarrier

calc_scatter_index = _get_flux_member("calc_scatter_index")

ProfilingContext = flux_mod.ProfilingContext
TuningRecord = flux_mod.TuningRecord
DistEnvTP = flux_mod.DistEnvTP
DistEnvTPWithEP = flux_mod.DistEnvTPWithEP
MoeArguments = flux_mod.MoeArguments

# GEMM only
GemmOnly = _get_flux_member("GemmOnly")
BlockScaleGemm = _get_flux_member("BlockScaleGemm")
GemmGroupedV2 = _get_flux_member("GemmGroupedV2")
GemmGroupedV3 = _get_flux_member("GemmGroupedV3")

# GEMM+RS
GemmRS = _get_flux_member("GemmRS")
get_gemm_rs_threadblock_segments_info = _get_flux_member("get_gemm_rs_threadblock_segments_info")
calc_gemm_rs_threadblock_segments_info = _get_flux_member("calc_gemm_rs_threadblock_segments_info")
GemmRSAcrossNode = _get_flux_member("GemmRSAcrossNode")

# allgather op
AGRingMode = _get_flux_member("AGRingMode")
AllGatherOption = _get_flux_member("AllGatherOption")
if not isinstance(AllGatherOption, NotCompiled):
    AllGatherOption.__repr__ = (
        lambda x: f"AllGatherOption(mode={x.mode}, fuse_sync={x.fuse_sync}, use_cuda_core_local={x.use_cuda_core_local}, use_cuda_core_ag={x.use_cuda_core_ag}, input_buffer_copied={x.input_buffer_copied}, use_read={x.use_read})"
    )
AllGatherOp = _get_flux_member("AllGatherOp")
RingMode = _get_flux_member("RingMode")
ReduceScatterOption = _get_flux_member("ReduceScatterOption")
if not isinstance(ReduceScatterOption, NotCompiled):
    ReduceScatterOption.__repr__ = (
        lambda x: f"use_barrier_queue={x.use_barrier_queue}, "
        "use_1d_ring={x.use_1d_ring}, "
        "use_p2p_read={x.use_p2p_read}, "
        "use_cudaMemcpyAsync={x.use_cudaMemcpyAsync}, "
        "use_gemmk={x.use_gemmk}, "
        "per_tile_flags={x.per_tile_flags}, "
        "n_split={x.n_split}, "
        "num_blocks={x.num_blocks}, "
        "ring_mode={x.ring_mode}"
    )

# AG+GEMM
AGKernel = _get_flux_member("AGKernel")
AGKernelCrossNode = _get_flux_member("AGKernelCrossNode")

# MOE ag-scatter
GemmGroupedV2AGScatterOp = _get_flux_member("GemmGroupedV2AGScatterOp")
GemmGroupedV3AGScatter = _get_flux_member("GemmGroupedV3AGScatter")
prepare_moe_ag_scatter_args = _get_flux_member("prepare_moe_ag_scatter_args")

# MOE gather-rs
GemmGroupedV2GatherRSOp = _get_flux_member("GemmGroupedV2GatherRSOp")
TopkReduceScatterOp = _get_flux_member("TopkReduceScatterOp")
GemmGroupedV3GatherRS = _get_flux_member("GemmGroupedV3GatherRS")
topk_scatter_reduce = _get_flux_member("topk_scatter_reduce")
All2AllOp = _get_flux_member("All2AllOp")

__all__ = [
    "bsr_reduce",
    "init_flux_shm",
    "create_tensor_list",
    "GroupBarrier",
    "bitwise_check",
    "uniform_initialize",
    "topk_scatter_reduce",
    "load_tuning_record",
    "TuningRecord",
    "ProfilingContext",
    "DistEnvTP",
    "DistEnvTPWithEP",
    "MoeArguments",
    "ReduceScatterOption",
    "RingMode",
    "GemmRS",
    "GemmRSAcrossNode",
    "AGKernel",
    "AGKernelCrossNode",
    "All2AllOp",
    "GemmOnly",
    "BlockScaleGemm",
    "GemmGroupedV2",
    "GemmGroupedV3",
    "GemmGroupedV3AGScatter",
    "GemmGroupedV3GatherRS",
    "GemmGroupedV2AGScatterOp",
    "prepare_moe_ag_scatter_args",
    "GemmGroupedV2GatherRSOp",
    "TopkReduceScatterOp",
    "AGRingMode",
    "AllGatherOption",
    "AllGatherOp",
    "get_gemm_rs_threadblock_segments_info",
    "calc_gemm_rs_threadblock_segments_info",
    "calc_scatter_index",
]
