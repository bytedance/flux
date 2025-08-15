#!/bin/bash

# set cuda env for scm
export PATH=/usr/local/cuda/bin:$PATH
export LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:$LIBRARY_PATH

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTORCH_24="2.4.0"
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
CLEAN_VERSION=$(echo "$PYTORCH_VERSION" | cut -d'+' -f1)

version_greater_or_eq() {
    # Compare two version numbers (greater than or equal)
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" = "$2" ]
}

if version_ge "$CLEAN_VERSION" "$PYTORCH_24"; then
    eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate flux
fi

git submodule init
git submodule update
pip install packaging
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo $NVCC_VERSION
NVCC_MAJOR_VERSION=$(echo $NVCC_VERSION | cut -d'.' -f1)
NVCC_MINOR_VERSION=$(echo $NVCC_VERSION | cut -d'.' -f2)

if [ "$NVCC_MAJOR_VERSION" -ge 12 ] && [ "$NVCC_MINOR_VERSION" -ge 4 ]; then
    ARCHS="80;89;90"
    SM_CORES="108;92;78;132"
elif [ "$NVCC_MAJOR_VERSION" -ge 12 ]; then
    ARCHS="80;90"
    SM_CORES="108;78;132"
else
    ARCHS="80"
    SM_CORES="108"
fi
echo $ARCHS
echo $SM_CORES

# adapt to scm envs that must start with CUSTOM_
# allow CUSTOM_JOBS to replace JOBS

export $(env | grep '^CUSTOM_JOBS' | sed 's/^CUSTOM_JOBS/JOBS/g')
./build.sh --arch ${ARCHS} --sm-cores $SM_CORES --nvshmem --package

cd $SCRIPT_DIR
mkdir -p output/python
unzip dist/*.whl -d output/python
