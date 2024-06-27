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

from contextlib import contextmanager
import ctypes
import os
import sys
from pathlib import Path
import torch
import importlib
import logging

FLUX_TORCH_EXTENSION_NAME = "flux_ths_pybind"


def _preload_libs(libname):
    libpath = Path(__file__).parent.parent / "lib" / libname
    try:
        ctypes.CDLL(libpath)
    except OSError as e:
        # Try to load from the LD_LIBRARY_PATH
        ctypes.CDLL(libname)


def _load_deps():
    _preload_libs("libflux_cuda.so")
    try:
        _preload_libs("nvshmem_bootstrap_torch.so")
        _preload_libs("nvshmem_transport_ibrc.so.2")
    except Exception as e:
        logging.info("Failed to load NVSHMEM libs")


_load_deps()
flux_mod = importlib.import_module(FLUX_TORCH_EXTENSION_NAME)


class NotCompiled:
    pass


bsr_reduce = getattr(flux_mod, "bsr_reduce", NotCompiled())
init_flux_shm = flux_mod.init_flux_shm
nvshmem_create_tensor = getattr(flux_mod, "nvshmem_create_tensor", NotCompiled())
bitwise_check = flux_mod.bitwise_check
uniform_initialize = flux_mod.uniform_initialize
load_tuning_record = flux_mod.load_tuning_record

ProfilingContext = flux_mod.ProfilingContext
TuningRecord = flux_mod.TuningRecord
DistEnvTP = flux_mod.DistEnvTP
DistEnvTPWithEP = flux_mod.DistEnvTPWithEP
MoeArguments = flux_mod.MoeArguments
GemmOnly = getattr(flux_mod, "GemmOnly", NotCompiled())
GemmRS = getattr(flux_mod, "GemmRS", NotCompiled())
GemmRSAcrossNode = getattr(flux_mod, "GemmRSAcrossNode", NotCompiled())
AllGatherGemm = getattr(flux_mod, "AllGatherGemm", NotCompiled())
AGKernel = getattr(flux_mod, "AGKernel", NotCompiled())
AGKernelCrossNode = getattr(flux_mod, "AGKernelCrossNode", NotCompiled())
AGKernelCrossNodeNvshmem = getattr(flux_mod, "AGKernelCrossNodeNvshmem", NotCompiled())
GemmGroupedV3AGScatter = getattr(flux_mod, "GemmGroupedV3AGScatter", NotCompiled())
GemmGroupedV3GatherRS = getattr(flux_mod, "GemmGroupedV3GatherRS", NotCompiled())
AgRingMode = getattr(flux_mod, "AGRingMode", NotCompiled())


def get_from_torch_classes(name: str):
    try:
        return getattr(torch.classes.flux, name)
    except:
        return NotCompiled()


All2AllOp = get_from_torch_classes("All2AllOp")
GemmGrouped = get_from_torch_classes("GemmGrouped")
GemmGroupedV3 = get_from_torch_classes("GemmGroupedV3")
# GemmGroupedV3GatherRS = get_from_torch_classes("GemmGroupedV3GatherRS")


__all__ = [
    "bsr_reduce",
    "init_flux_shm",
    "bitwise_check",
    "uniform_initialize",
    "load_tuning_record",
    "TuningRecord",
    "ProfilingContext",
    "DistEnvTP",
    "DistEnvTPWithEP",
    "MoeArguments",
    "GemmOnly",
    "GemmRS",
    "GemmRSAcrossNode",
    "AllGatherGemm",
    "AGKernel",
    "AGKernelCrossNode",
    "All2AllOp",
    "GemmGrouped",
    "GemmGroupedV3",
    "GemmGroupedV3AGScatter",
    "GemmGroupedV3GatherRS",
    "AgRingMode",
    "nvshmem_create_tensor",
]
