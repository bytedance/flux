import glob
import os
import shutil
import sys
import re
import ast
from pathlib import Path
import urllib
import urllib.request
import urllib.error
import setuptools
import torch
import subprocess
from torch.utils.cpp_extension import BuildExtension
from packaging.version import parse, Version
from typing import Optional, Tuple
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

# Project directory root
root_path: Path = Path(__file__).resolve().parent
enable_nvshmem = int(os.getenv("FLUX_SHM_USE_NVSHMEM", 0))
PACKAGE_NAME = "byte_flux"
BASE_WHEEL_URL = "https://github.com/bytedance/flux/releases/download/{tag_name}/{wheel_name}"
FORCE_BUILD = os.getenv("FLASH_ATTENTION_FORCE_BUILD", "FALSE") == "TRUE"
USE_LOCAL_VERSION = int(os.getenv("FLUX_USE_LOCAL_VERSION", 0))

def cuda_version() -> Tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) by nvcc --version"""

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

    # Query NVCC for version info
    output = subprocess.run(
        [nvcc_bin, "-V"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    version = match.group(1).split(".")
    return tuple(int(v) for v in version)


def get_local_version(public_version):
    cuda_version_major, cuda_version_minor = cuda_version()
    torch_version_splits = torch.__version__.split(".")
    torch_version = f"{torch_version_splits[0]}.{torch_version_splits[1]}"
    version = public_version + f"+cu{cuda_version_major}{cuda_version_minor}" + f"torch{torch_version}"
    return version

def get_public_version():
    with open(Path(root_path) / "python" / "flux" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    return public_version

def get_package_version():
    global USE_LOCAL_VERSION
    public_version = get_public_version()
    if USE_LOCAL_VERSION:
        return get_local_version(public_version)
    return public_version

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
    if not os.path.exists(file_path):
        # flux is installed through pip3, the flux_ths_files.txt is not generated
        return []
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

    deps = [nccl_deps(), cutlass_deps(), flux_cuda_deps(), cuda_deps()]
    if enable_nvshmem:
        deps.append(nvshmem_deps())
    for include_dir, library_dir, library in deps:
        include_dirs += include_dir
        library_dirs += library_dir
        libraries += library

    # Compiler flags
    # too much warning from CUDA /usr/local/cuda/include/cusparse.h: "-Wdeprecated-declarations"
    cxx_flags = [
        "-O3",
        "-DTORCH_CUDA=1",
        "-fvisibility=hidden",
        "-Wno-deprecated-declarations",
        "-fdiagnostics-color=always",
    ]
    if enable_nvshmem:
        cxx_flags.append("-DFLUX_SHM_USE_NVSHMEM")
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


def get_wheel_url():
    flux_tag_version = get_public_version()
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_version_raw = parse(torch.__version__)
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    torch_cuda_version = parse(torch.version.cuda)
    torch_cuda_version = parse("11.8") if torch_cuda_version.major == 11 else parse("12.3")
    cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
    wheel_filename = f"{PACKAGE_NAME}-{flux_tag_version}+cu{cuda_version}torch{torch_version}-{python_version}-{python_version}-linux_x86_64.whl"
    wheel_url = BASE_WHEEL_URL.format(tag_name=f"v{flux_tag_version}", wheel_name=wheel_filename)
    return wheel_url, wheel_filename


class CachedWheelsCommand(_bdist_wheel):
    """
    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip when it cannot
    find an existing wheel (which is currently the case for all flash attention installs). We use
    the environment parameters to detect whether there is already a pre-built version of a compatible
    wheel available and short-circuits the standard full build pipeline.
    """

    def run(self):
        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            shutil.move(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


def main():
    flux_version = get_package_version()
    packages = setuptools.find_packages(
        where="python",
        include=["flux", "flux.pynvshmem", "flux_ths_pybind"],
    )
    data_file_list = ["python/lib/libflux_cuda.so"]
    if enable_nvshmem:
        data_file_list += [
            "python/lib/nvshmem_bootstrap_torch.so",
            "python/lib/nvshmem_transport_ibrc.so.2",
            "python/lib/libnvshmem_host.so.2",
        ]
    # Configure package
    setuptools.setup(
        name=PACKAGE_NAME,
        version=flux_version,
        package_dir={"": "python"},
        packages=packages,
        description="Flux library",
        ext_modules=[setup_pytorch_extension()],
        cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension},
        setup_requires=["torch", "cmake", "packaging"],
        install_requires=["torch"],
        extras_require={"test": ["torch", "numpy"]},
        license_files=("LICENSE",),
        package_data={"python/lib": ["*.so"]},  # only works for sdist
        python_requires=">=3.8",
        # include_package_data=True,
        data_files=[
            (
                "lib",  # installed directory
                data_file_list,  # to installed shared libraries. only works for setup.py install/bdist_wheels
            )
        ],
    )


if __name__ == "__main__":
    main()
