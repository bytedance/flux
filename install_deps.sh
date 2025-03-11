#!/bin/bash
script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir" 

# Patch CUTLASS
cd 3rdparty/cutlass && git checkout v3.7.0 && cd ..
patch -p1 < ./cutlass3.7.patch

# Download NVSHMEM
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
tar zxvf nvshmem_src_3.2.5-1.txz
mv nvshmem_src nvshmem
rm nvshmem_src_3.2.5-1.txz