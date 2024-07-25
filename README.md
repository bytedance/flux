# Flux

Flux is a fast communication-overlapping library for tensor parallelism on GPUs.


## Why Flux

Flux can significantly reduce latency and increase throughput for tensor parallelism for both inference and training.

## Install from pip
```
# Make sure that PyTorch is installed.
pip install packaging
pip install byte-flux
```

## Build from source
```bash
git clone https://github.com/bytedance/flux.git
git submodule update --init --recursive
# Ampere
./build.sh --arch 80 
# Hopper
./build.sh --arch 90 
```
## Build for cross-machine TP
FLUX relies on NVSHMEM for communication across nodes. Therefore, if you need support for cross-machine tensor parallelism (TP), you must manually download the NVSHMEM source code and enable the nvshmem option during compilation.

```bash
git clone https://github.com/bytedance/flux.git
# Download nvshmem-2.11(https://developer.nvidia.com/nvshmem) and place it to flux/3rdparty/nvshmem
# Flux is temporarily dependent on a specific version of nvshmem (2.11).
tar Jxvf nvshmem_src_2.11.0-5.txz
mv nvshmem_src_2.11.0-5 ${YOUR_PATH}/flux/3rdparty/nvshmem
git submodule update --init --recursive

# Ampere
./build.sh --arch 80 --nvshmem
# Hopper
./build.sh --arch 90 --nvshmem
```

If you are tired of the cmake process, you can set environment variable `FLUX_BUILD_SKIP_CMAKE` to 1 to skip cmake if `build/CMakeCache.txt` already exists.

If you want to build a wheel package, add `--package` to the build command. find the output wheel file under dist/

```bash
# Ampere
./build.sh --arch 80 --package

# Hopper
./build.sh --arch 90 --package
```


## Run Demo
```bash
# gemm only
PYTHONPATH=./python:$PYTHONPATH python3 test/test_gemm_only.py 4096 12288 6144 --dtype=float16

# gemm fused with reduce-scatter
./scripts/launch.sh test/test_gemm_rs.py 4096 12288 49152 --dtype=float16 --iters=10

# all-gather fused with gemm
./scripts/launch.sh test/test_ag_kernel.py 4096 49152 12288 --dtype=float16 --iters=10
```

## Performance
We measured the examples from the above demo on both A800s and H800s. Each machine has 8 GPUs, with a TP size set to 8. The table below shows the performance comparison between flux and torch+nccl. It can be observed that by overlapping fine-grained computation and communication, Flux is able to effectively hide a significant portion of the communication time

|  | M | K | N | Torch Gemm | Torch NCCL | Torch Total | Flux Gemm | Flux Comm | Flux Total |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|
| AG+Gemm(A800) | 4096 | 12288 | 49152 | 2.438ms | 0.662ms | 3.099ms | 2.378ms | 0.091ms | 2.469ms |
| Gemm+RS(A800) | 4096 | 49152 | 12288 | 2.453ms | 0.646ms | 3.100ms | 2.429ms | 0.080ms | 2.508ms |
| AG+Gemm(H800) | 4096 | 12288 | 49152 | 0.846ms | 0.583ms | 1.429ms | 0.814ms | 0.143ms | 0.957ms |
| Gemm+RS(H800) | 4096 | 49152 | 12288 | 0.818ms | 0.590ms | 1.408ms | 0.822ms | 0.111ms | 0.932ms |

AG refers to AllGather.
RS refers to ReduceScatter.


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
```

## Reference

* [ArXiv Paper](http://arxiv.org/abs/2406.06858)

## [License](./LICENSE)

The Flux Project is under the Apache License v2.0.
