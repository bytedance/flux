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

import glob
import os
import re
from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension

from gen_version import generate_versoin_file, check_final_release

# Project directory root
root_path: Path = Path(__file__).resolve().parent

version_txt = os.path.join(root_path, "version.txt")
version_file = os.path.join(root_path, "python/flux/version.py")
is_dev = not check_final_release()
flux_version = generate_versoin_file(version_txt, version_file, dev=is_dev)


def pathlib_wrapper(func):
    def wrapper(*kargs, **kwargs):
        include_dirs, library_dirs, libraries = func(*kargs, **kwargs)
        return map(str, include_dirs), map(str, library_dirs), map(str, libraries)

    return wrapper


@pathlib_wrapper
def cutlass_deps():
    cutlass_home = root_path / "3rdparty/cutlass"
    include_dirs = [
        cutlass_home / "include",
        cutlass_home / "tools" / "util" / "include",
        cutlass_home / "tools" / "library" / "include",
        cutlass_home / "tools" / "profiler" / "include",
    ]
    library_dirs = []
    libraries = []
    return include_dirs, library_dirs, libraries


def read_flux_ths_files():
    file_path = root_path / "build/src/ths_op/flux_ths_files.txt"
    variables = {}
    with open(file_path, "r") as file:
        for line in file:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                variables[key] = value
    return variables["FLUX_THS_FILES"].split(";")


@pathlib_wrapper
def nvshmem_deps():
    nvshmem_home = Path(os.environ.get("NVSHMEM_HOME", root_path / "3rdparty/nvshmem/build/src"))
    include_dirs = [nvshmem_home / "include"]
    library_dirs = [nvshmem_home / "lib"]
    # libraries = ["nvshmem"]
    libraries = ["nvshmem_host"]
    return include_dirs, library_dirs, libraries


@pathlib_wrapper
def flux_cuda_deps():
    include_dirs = [root_path / "include", root_path / "src"]
    library_dirs = [root_path / "build" / "lib"]
    libraries = ["flux_cuda"]
    return include_dirs, library_dirs, libraries


@pathlib_wrapper
def cuda_deps():
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    include_dirs = [cuda_home / "include"]
    library_dirs = [cuda_home / "lib64", cuda_home / "lib64/stubs"]
    libraries = ["cuda", "cudart", "nvidia-ml"]
    return include_dirs, library_dirs, libraries


@pathlib_wrapper
def nccl_deps():
    nccl_home = Path(os.environ.get("NCCL_ROOT", root_path / "3rdparty/nccl/build/local"))
    include_dirs = [nccl_home / "include", nccl_home / "include" / "nccl" / "detail" / "include"]
    library_dirs = [nccl_home / "lib"]
    libraries = ["nccl_static"]
    return include_dirs, library_dirs, libraries


def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CppExtension for PyTorch support"""
    include_dirs, library_dirs, libraries = [], [], []
    for include_dir, library_dir, library in (
        nccl_deps(),
        cutlass_deps(),
        flux_cuda_deps(),
        nvshmem_deps(),
        cuda_deps(),
    ):
        include_dirs += include_dir
        library_dirs += library_dir
        libraries += library

    # Compiler flags
    # too much warning from CUDA /usr/local/cuda/include/cusparse.h: "-Wdeprecated-declarations"
    cxx_flags = ["-O3", "-DTORCH_CUDA=1", "-fvisibility=hidden", "-Wno-deprecated-declarations"]
    ld_flags = ["-Wl,--exclude-libs=libnccl_static"]
    flux_ths_files = read_flux_ths_files()
    from torch.utils.cpp_extension import CppExtension

    return CppExtension(
        name="flux_ths_pybind",
        sources=flux_ths_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=cxx_flags,
        extra_link_args=ld_flags,
    )


def main():
    # Submodules to install
    packages = setuptools.find_packages(
        where="python",
        include=["flux", "flux.pynvshmem", "flux_ths_pybind"],
    )

    # Configure package
    setuptools.setup(
        name="flux",
        version=flux_version,
        package_dir={"": "python"},
        packages=packages,
        description="Flux library",
        ext_modules=[setup_pytorch_extension()],
        cmdclass={"build_ext": BuildExtension},
        setup_requires=["torch", "cmake"],
        install_requires=["torch"],
        extras_require={"test": ["torch", "numpy"]},
        license_files=("LICENSE",),
        package_data={"python/lib": ["*.so"]},  # only works for sdist
        # include_package_data=True,
        data_files=[
            (
                "lib",  # installed directory
                [
                    "python/lib/nvshmem_bootstrap_torch.so",
                    "python/lib/libflux_cuda.so",
                    "python/lib/nvshmem_transport_ibrc.so.2",
                    "python/lib/libnvshmem_host.so.2",
                ],  # to installed shared libraries. only works for setup.py install/bdist_wheels
            )
        ],
    )


if __name__ == "__main__":
    main()
