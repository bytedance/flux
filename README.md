<div align="center">
 👋 Hi, everyone!
    <br>
    We are <b>ByteDance Seed team.</b>
</div>

<p align="center">
  You can get to know us better through the following channels👇
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

# Flux: Fine-grained Computation-communication Overlapping GPU Kernel Library

<p align="center">
  <a href="https://github.com/bytedance/flux">
    <img src="https://img.shields.io/badge/Flux-Project Page-yellow"></a>
  <a href="https://arxiv.org/pdf/2502.19811">
    <img src="https://img.shields.io/badge/Flux-Tech Report-red"></a>
  <a href="XXX">
    <img src="https://img.shields.io/badge/License-Apache-blue"></a>
</p>

Flux is a communication-overlapping library for dense/MoE models on GPUs, providing high-performance and pluggable kernels to support various parallelisms in model training/inference.

Flux's efficient kernels are compatible with Pytorch and can be integrated into existing frameworks easily, supporting various Nvidia GPU architectures and data types.

# News
[2025/03/10]🔥We have released **COMET: Computation-communication Overlapping for Mixture-of-Experts**.


## Getting started
Install Flux from source.

### Install from Source
```bash
git clone --recursive https://github.com/bytedance/flux.git && cd flux

# Install dependencies
bash ./install_deps.sh

# For Ampere(sm80) GPU
./build.sh --arch 80 --nvshmem
# For Ada Lovelace(sm89) GPU
./build.sh --arch 89 --nvshmem
# For Hopper(sm90) GPU
./build.sh --arch 90 --nvshmem
```

#### Install in a virtual environment
Here is a snippet to install Flux in a virtual environment. Let's finish the installation in an virtual environment with CUDA 12.4, torch 2.6.0 and python 3.11.

```bash
conda create -n flux python=3.11
conda activate flux
pip3 install packaging
pip3 install ninja
pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

./build.sh --clean-all
./build.sh --arch "80;89;90" --nvshmem --package
```

Then you would expect a wheel package under `dist/` folder that is suitable for your virtual environment.

### Customized Installation
#### Build options for source installation

1. Add `--nvshmem` to build Flux with NVSHMEM support. It is essential for the MoE kernels.
2. If you are tired of the cmake process, you can set environment variable `FLUX_BUILD_SKIP_CMAKE` to 1 to skip cmake if `build/CMakeCache.txt` already exists.
3. If you want to build a wheel package, add `--package` to the build command. find the output wheel file under dist/


#### Dependencies
The core dependencies of Flux are NCCL, CUTLASS, and NVSHMEM, which are located under the 3rdparty folder.
1. NCCL: Managed by git submodule automatically.
2. NVSHMEM: Downloaded from https://developer.nvidia.com/nvshmem. The current version is 3.2.5-1.
3. CUTLASS: Flux leverages CUTLASS to generate high-performance GEMM kernels. We currently use CUTLASS 3.7.0 and a tiny patch is applied to CUTLASS.


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

* For a more detailed usage on MoE kernels, please refer to [Flux MoE Usage](https://github.com/bytedance/flux/blob/main/docs/moe_usage.md). Try some [examples](https://github.com/bytedance/flux/blob/main/examples) as a quick start. A [minimal MoE layer](https://github.com/bytedance/flux/blob/main/examples/moe_flux_only.py) can be implemented within only a few tens of lines of code using Flux!
* For some performance numbers, please refer to [Performance Doc](https://github.com/bytedance/flux/blob/main/docs/performance.md).
* To learn more about the design principles of Flux, please refer to [Design Doc](https://github.com/bytedance/flux/blob/main/docs/design.md).


## [License](./LICENSE)

The Flux Project is under the Apache License v2.0.

## Citation

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


# About [ByteDance Seed Team](https://team.doubao.com/)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

