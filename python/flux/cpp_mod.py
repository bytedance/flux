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

FLUX_TORCH_EXTENSION_NAME = "flux_ths_pybind"


def _preload_libs(libname):
    libpath = Path(__file__).parent.parent / "lib" / libname
    try:
        ctypes.CDLL(libpath)
    except OSError as e:
        # print(f"error load {libname} from {libpath}... try load from LD_LIBRARY_PATH...")
        ctypes.CDLL(libname)


def _load_deps():
    _preload_libs("libflux_cuda.so")
    _preload_libs("nvshmem_bootstrap_torch.so")
    _preload_libs("nvshmem_transport_ibrc.so.2")


_load_deps()
flux_mod = importlib.import_module(FLUX_TORCH_EXTENSION_NAME)


class NotCompiled:
    pass


bsr_reduce = getattr(flux_mod, "bsr_reduce", NotCompiled())
bitwise_check = flux_mod.bitwise_check
uniform_initialize = flux_mod.uniform_initialize
load_tuning_record = flux_mod.load_tuning_record

ProfilingContext = flux_mod.ProfilingContext
TuningRecord = flux_mod.TuningRecord
DistEnvTP = flux_mod.DistEnvTP
DistEnvTPWithEP = flux_mod.DistEnvTPWithEP

GemmOnly = getattr(flux_mod, "GemmOnly", NotCompiled())
GemmRS = getattr(flux_mod, "GemmRS", NotCompiled())
GemmRSAcrossNode = getattr(flux_mod, "GemmRSAcrossNode", NotCompiled())
AllGatherGemm = getattr(flux_mod, "AllGatherGemm", NotCompiled())
AGKernel = getattr(flux_mod, "AGKernel", NotCompiled())
AGKernelCrossNode = getattr(flux_mod, "AGKernelCrossNode", NotCompiled())
AGKernelCrossNodeNvshmem = getattr(flux_mod, "AGKernelCrossNodeNvshmem", NotCompiled())
AgRingMode = getattr(flux_mod, "AGRingMode", NotCompiled())


def get_from_torch_classes(name: str):
    try:
        return getattr(torch.classes.flux, name)
    except:
        return NotCompiled()


pynvshmem_mod = flux_mod._pynvshmem

__all__ = [
    "bsr_reduce",
    "bitwise_check",
    "uniform_initialize",
    "load_tuning_record",
    "TuningRecord",
    "ProfilingContext",
    "DistEnvTP",
    "DistEnvTPWithEP",
    "GemmOnly",
    "GemmRS",
    "GemmRSAcrossNode",
    "AllGatherGemm",
    "AGKernel",
    "AGKernelCrossNode",
    "pynvshmem_mod",
    "AgRingMode",
]
