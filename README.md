# Flux

Flux is a fast communication-overlapping library for tensor parallelism on GPUs.


## Why Flux

Flux significantly can reduce latency and increase throughput for tensor parallelism for both inference and training.

## Build
```bash
# Download nvshmem-2.11(https://developer.nvidia.com/nvshmem) and place it to flux/3rdparty/nvshmem
# Flux is temporarily dependent on a specific version of nvshmem (2.11).
tar Jxvf nvshmem_src_2.11.0-5.txz
mv nvshmem_src_2.11.0-5 ${YOUR_PATH}/flux/3rdparty/nvshmem
git submodule update --init --recursive
# Ampere
./build.sh --arch 80

```

If you are tired of the cmake process, you can set environment variable `FLUX_BUILD_SKIP_CMAKE` to 1 to skip cmake if `build/CMakeCache.txt` already exists.

If you want to build a wheel package, add `--package` to the build command. find the output wheel file under dist/

```
# Ampere
./build.sh --arch 80 --package
```

For development release, run build script with `FLUX_FINAL_RELEASE=0`.

```
# Ampere
FLUX_FINAL_RELEASE=0 ./build.sh --arch 80 --package
```

## Run Demo
```
# gemm only
PYTHONPATH=./python:$PYTHONPATH python3 test/test_gemm_only.py 4096 12288 6144 --dtype=float16

# gemm fused with reduce-scatter
./scripts/launch.sh test/test_gemm_rs.py 4096 12288 49152 --dtype=float16 --iters=10

# all-gather fused with gemm
./scripts/launch.sh test/test_ag_kernel.py 4096 49152 12288 --dtype=float16 --iters=10
```



## [License](./LICENSE)

The Flux Project is under the Apache License v2.0.
