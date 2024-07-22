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

import argparse
import os
import subprocess
from pathlib import Path
import shutil
import re
from typing import Optional, Tuple
import torch

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


def _check_env_option(opt, default=""):
    return os.getenv(opt, default).upper() in ["ON", "1", "YES", "TRUE"]


def check_final_release():
    return _check_env_option("FLUX_FINAL_RELEASE", "1")


def get_git_commit(src_dir):
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src_dir)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


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


def get_flux_version(version_txt, *, dev=False):
    with open(version_txt) as f:
        version = f.readline()
        version = version.strip()
    cuda_version_major, cuda_version_minor = cuda_version()
    torch_version_splits = torch.__version__.split(".")
    torch_version = f"{torch_version_splits[0]}.{torch_version_splits[1]}"
    version = version + f"+cu{cuda_version_major}{cuda_version_minor}" + f"torch{torch_version}"
    if dev:
        commit_id = get_git_commit(CUR_DIR)

        version += ".dev{}".format(commit_id[:8])
    # version = version + (f'.{os.getenv("ARCH")}' if os.getenv("ARCH") else "")
    return version


def generate_versoin_file(version_txt, version_file, *, dev=False):
    flux_ver = get_flux_version(version_txt, dev=dev)

    with open(version_file, "w") as f:
        f.write("__version__ = '{}'\n".format(flux_ver))
        f.write("git_version = {}\n".format(repr(get_git_commit(CUR_DIR))))
        cuda_version_major, cuda_version_minor = cuda_version()
        f.write("cuda = {}.{}\n".format(cuda_version_major, cuda_version_minor))

    return flux_ver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate version.py")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    generate_versoin_file(args.input, args.output, dev=args.dev)
