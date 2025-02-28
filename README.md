# Flux

Flux is a fast communication-overlapping library for dense/MoE models on GPUs, providing high-performance and pluggable kernels to support various parallelisms in model training/inference.

## Installation
Install Flux either from PyPI or from source.

### PyPI

```
# Make sure that PyTorch is installed.
pip install packaging
pip install byte-flux
```

### Build from source
```bash
git clone https://github.com/bytedance/flux.git
git submodule update --init --recursive
# Ampere
./build.sh --arch 80 
# Hopper
./build.sh --arch 90 
```

If you are tired of the cmake process, you can set environment variable `FLUX_BUILD_SKIP_CMAKE` to 1 to skip cmake if `build/CMakeCache.txt` already exists.

If you want to build a wheel package, add `--package` to the build command. find the output wheel file under dist/

```bash
# Ampere
./build.sh --arch 80 --package

# Hopper
./build.sh --arch 90 --package
```

### Dependencies

#### NVSHMEM
The current nvshmem folder under flux/3rdparty is extracted from the nvshmem_src_3.1.7-1.txz downloaded from https://developer.nvidia.com/nvshmem. Users can also try other newer versions of nvshmem.

#### CUTLASS
Flux leverages CUTLASS to generate high-performance GEMM kernels. We currently use CUTLASS 3.7.0 and a tiny patch should be applied to CUTLASS.

```
cd 3rdparty
git clone https://github.com/NVIDIA/cutlass
git checkout v3.7.0
patch -p1 < ../cutlass3.7.patch
```

#### NCCL
Managed by git submodule automatically.


## Run Demo
```bash
# gemm only
PYTHONPATH=./python:$PYTHONPATH python3 test/test_gemm_only.py 4096 12288 6144 --dtype=float16

# gemm fused with reduce-scatter
./scripts/launch.sh test/test_gemm_rs.py 4096 12288 49152 --dtype=float16 --iters=10

# all-gather fused with gemm
./scripts/launch.sh test/test_ag_kernel.py 4096 49152 12288 --dtype=float16 --iters=10
```



## Citing

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

* [ArXiv Paper](http://arxiv.org/abs/2406.06858)

## [License](./LICENSE)

The Flux Project is under the Apache License v2.0.
