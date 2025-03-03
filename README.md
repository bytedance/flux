# Flux

Flux is a communication-overlapping library for dense/MoE models on GPUs, providing high-performance and pluggable kernels to support various parallelisms in model training/inference.

Flux's efficient kernels are compatible with Pytorch and can be integrated into existing frameworks easily, supporting various Nvidia GPU architectures and data types.

## Installation
Install Flux either from PyPI or from source.

### Install from PyPI

```
# Make sure that PyTorch is installed.
pip install packaging
pip install byte-flux
```

### Build from source
```bash
git clone --recursive https://github.com/bytedance/flux.git
git checkout comet

# Patch CUTLASS
cd 3rdparty/cutlass
git checkout v3.7.0
cd ..
patch -p1 < ./cutlass3.7.patch

# Ampere
./build.sh --arch 80 --nvshmem
# Ada Lovelace
./build.sh --arch 89 --nvshmem
# Hopper
./build.sh --arch 90 --nvshmem
```

#### Build in a virtual environment
Here is a snippet to install Flux in a virtual environment. Let's finish the installation in an virtual environment with CUDA 12.4, torch 2.5.0 and python 3.11.

```bash
conda create -n flux python=3.11
conda activate flux
pip3 install packaging
pip3 install torch==2.5.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

./build.sh --clean-all
./build.sh --arch "80;89;90" --nvshmem --package
```


#### Build options

1. Add `--nvshmem` to build Flux with NVSHMEM support. It is essential for the MoE kernels.
2. If you are tired of the cmake process, you can set environment variable `FLUX_BUILD_SKIP_CMAKE` to 1 to skip cmake if `build/CMakeCache.txt` already exists.
3. If you want to build a wheel package, add `--package` to the build command. find the output wheel file under dist/

```bash
# Ampere
./build.sh --arch 80 --package
# Ada Lovelace
./build.sh --arch 89 --package
# Hopper
./build.sh --arch 90 --package
```

#### Dependencies
The core dependencies of Flux are NCCL, CUTLASS, and NVSHMEM, which are located under the 3rdparty folder.
1. NCCL: Managed by git submodule automatically.
2. NVSHMEM: The current nvshmem folder under flux/3rdparty is extracted from the nvshmem_src_3.1.7-1.txz downloaded from https://developer.nvidia.com/nvshmem. Users can also try other newer versions of nvshmem.
3. CUTLASS: Flux leverages CUTLASS to generate high-performance GEMM kernels. We currently use CUTLASS 3.7.0 and a tiny patch should be applied to CUTLASS.


## Quick Start

Below are commands to run some basic demos once you have installed Flux successfully.
```bash
# gemm only
python3 test/python/gemm_only/test_gemm_only.py 4096 12288 6144 --dtype=float16

# all-gather fused with gemm (dense MLP layer0)
./launch.sh test/python/ag_gemm/test_ag_kernel.py 4096 49152 12288 --dtype=float16 --iters=10

# gemm fused with reduce-scatter (dense MLP layer1)
./launch.sh test/python/gemm_rs/test_gemm_rs.py 4096 12288 49152 --dtype=float16 --iters=10

# all-gather fused with grouped gemm (MoE MLP layer0)
./launch.sh test/python/moe_ag_scatter/test_moe_ag.py

# grouped gemm fused with reduce-scatter (MoE MLP layer1)
./launch.sh test/python/moe_gather_rs/test_moe_gather_rs.py
```

You can check out the documentations for more details!
* For a more detailed usage on MoE kernels, please refer to [Flux MoE Doc](https://github.com/ZSL98/flux/blob/comet_clean/docs/mlsys_comet_ae.md).


## Citations

If you use Flux in a scientific publication, we encourage you to add the following reference
to the related papers:
```
@misc{chang2024flux,
      title={FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion},
      author={Li-Wen Chang and Wenlei Bao and Qi Hou and Chengquan Jiang and Ningxin Zheng and Yinmin Zhong and Xuanrun Zhang and Zuquan Song and Ziheng Jiang and Haibin Lin and Xin Jin and Xin Liu},
      year={2024},
      eprint={2406.06858},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{zhang2025comet,
      title={Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts},
      author={Shulai Zhang, Ningxin Zheng, Haibin Lin, Ziheng Jiang, Wenlei Bao, Chengquan Jiang, Qi Hou, Weihao Cui, Size Zheng, Li-Wen Chang, Quan Chen and Xin Liu},
      year={2025},
      eprint={2502.19811},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}

```

## Reference

* [ArXiv Paper (Flux)](http://arxiv.org/abs/2406.06858)
* [ArXiv Paper (Comet)](https://arxiv.org/abs/2502.19811)

## [License](./LICENSE)

The Flux Project is under the Apache License v2.0.
