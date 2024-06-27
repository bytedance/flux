#!/bin/bash
set -x
set -e

export PATH=/usr/local/cuda/bin:$PATH
CMAKE=${CMAKE:-cmake}

ARCH=""
BUILD_TEST="ON"
BDIST_WHEEL="OFF"
WITH_PROTOBUF="OFF"
FLUX_DEBUG="OFF"
ENABLE_NVSHMEM="OFF"

function clean_py() {
    rm -rf build/lib.*
    rm -rf python/lib
    rm -rf python/flux.egg-info
    rm -rf python/flux_ths_pybind.*
}

function clean_all() {
    clean_py
    rm -rf build/
    rm -rf pynvshmem/build/
    rm -rf 3rdparty/nvshmem/build
}

# Iterate over the command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --arch)
        # Process the arch argument
        ARCH="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --no_test)
        BUILD_TEST="OFF"
        shift # Skip the argument value
        ;;
    --jobs)
        # Process the jobs argument
        JOBS="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --clean-py)
        clean_py
        exit 0
        ;;
    --clean-all)
        clean_all
        exit 0
        ;;
    --debug)
        FLUX_DEBUG="ON"
        shift
        ;;
    --package)
        BDIST_WHEEL="ON"
        shift # Skip the argument key
        ;;
    --protobuf)
        WITH_PROTOBUF="ON"
        shift
        ;;
    --nvshmem)
        ENABLE_NVSHMEM="ON"
        shift
        ;;
    *)
        # Unknown argument
        echo "Unknown argument: $1"
        shift # Skip the argument
        ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${SCRIPT_DIR}
PROTOBUF_ROOT=$PROJECT_ROOT/3rdparty/protobuf

cd ${PROJECT_ROOT}

if [[ -n $ARCH ]]; then
    build_args=" --arch ${ARCH}"
fi

if [[ -z $JOBS ]]; then
    JOBS=$(nproc --ignore 2)
fi

##### build protobuf #####
function build_protobuf() {
    if [ $WITH_PROTOBUF == "ON" ]; then
        pushd $PROTOBUF_ROOT
        mkdir -p $PWD/build/local
        pushd build
        CFLAGS="-fPIC" CXXFLAGS="-fPIC" cmake ../cmake -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$(realpath local)
        make -j$(nproc)
        make install
        popd
        popd
    fi
}

function build_nccl() {
    pushd $NCCL_ROOT
    export BUILDDIR=${NCCL_ROOT}/build
    export PREFIX=${BUILDDIR}/local

    if [[ -n $ARCH ]]; then
        NCCL_COMPILE_OPTIONS_ARCH="" # default none
        arch_list=()
        IFS=";" read -ra arch_list <<<"$ARCH"
        for arch in "${arch_list[@]}"; do
            NCCL_COMPILE_OPTIONS_ARCH="-gencode=arch=compute_${arch},code=sm_${arch} ${NCCL_COMPILE_OPTIONS_ARCH}"
        done
        make -j${nproc} src.staticlib NVCC_GENCODE="${NCCL_COMPILE_OPTIONS_ARCH}" VERBOSE=1
    else
        make -j${nproc} src.staticlib VERBOSE=1
    fi
    # only install static lib
    mkdir -p ${PREFIX}/lib
    mkdir -p ${PREFIX}/include/nccl/detail
    cp -P -v ${BUILDDIR}/lib/lib* ${PREFIX}/lib/
    cp -v ${BUILDDIR}/include/* ${PREFIX}/include/nccl/detail
    pushd ${NCCL_ROOT}/src
    find . -type f -name "*.h" -print0 | xargs -0 cp --target-directory=${PREFIX}/include/nccl/detail --parents
    find . -type f -name "*.hpp" -print0 | xargs -0 cp --target-directory=${PREFIX}/include/nccl/detail --parents
    popd
    popd
}

##### build nvshmem_bootstrap_torch  #####
function build_pynvshmem() {
    PYNVSHMEM_DIR=$PROJECT_ROOT/pynvshmem
    export NVSHMEM_HOME=$PROJECT_ROOT/3rdparty/nvshmem/build/src
    mkdir -p ${PYNVSHMEM_DIR}/build

    pushd ${PYNVSHMEM_DIR}/build
    if [ ! -f CMakeCache.txt ] || [ -z ${FLUX_BUILD_SKIP_CMAKE} ]; then
        ${CMAKE} .. \
            -DNVSHMEM_HOME=${NVSHMEM_HOME} \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=1
    fi
    make -j nvshmem_bootstrap_torch
    popd
}

##### build flux_cuda #####
function build_flux_cuda() {
    mkdir -p build
    pushd build
    if [ ! -f CMakeCache.txt ] || [ -z ${FLUX_BUILD_SKIP_CMAKE} ]; then
        CMAKE_ARGS=(
            -DENABLE_NVSHMEM=${ENABLE_NVSHMEM}
            -DCUDAARCHS=${ARCH}
            -DCMAKE_EXPORT_COMPILE_COMMANDS=1
            -DBUILD_TEST=${BUILD_TEST}
        )
        if [ $WITH_PROTOBUF == "ON" ]; then
            CMAKE_ARGS+=(
                -DWITH_PROTOBUF=ON
                -DProtobuf_ROOT=${PROTOBUF_ROOT}/build/local
                -DProtobuf_PROTOC_EXECUTABLE=${PROTOBUF_ROOT}/build/local/bin/protoc
            )
        fi
        if [ $FLUX_DEBUG == "ON" ]; then
            CMAKE_ARGS+=(
                -DFLUX_DEBUG=ON
            )
        fi
        ${CMAKE} .. ${CMAKE_ARGS[@]}
    fi
    make -j${JOBS} VERBOSE=1
    popd
}

function merge_compile_commands() {
    if command -v ninja >/dev/null 2>&1; then
        # generate compile_commands.json
        ninja -f $(ls ./build/temp.*/build.ninja) -t compdb >build/compile_commands_ths_op.json
        cat >build/merge_compile_commands.py <<EOF
import json
with open("build/compile_commands.json") as f:
    cmds = json.load(f)
with open("build/compile_commands_ths_op.json") as f:
    cmds_ths_op = json.load(f)
with open("build/compile_commands.json", "w") as f:
    json.dump(cmds+cmds_ths_op, f, indent=2)
EOF

        python3 build/merge_compile_commands.py
        echo "merge compile_commands.json done"
    else
        echo "Ninja is not installed. Ninja is required for flux_ths_pybind's compile_commands.json. run 'pip3 install ninja'"
    fi
}

function build_flux_py {
    LIBDIR=${PROJECT_ROOT}/python/lib
    rm -rf ${LIBDIR}
    mkdir -p ${LIBDIR}

    # rm -f ${LIBDIR}/libflux_cuda.so
    # rm -f ${LIBDIR}/nvshmem_bootstrap_torch.so
    # rm -f ${LIBDIR}/nvshmem_transport_ibrc.so.2
    # rm -f ${LIBDIR}/libnvshmem_host.so.2
    pushd ${LIBDIR}
    cp -s ../../build/lib/libflux_cuda.so .
    if [ $ENABLE_NVSHMEM == "ON" ]; then
        cp -s ../../pynvshmem/build/nvshmem_bootstrap_torch.so .
        cp -s ../../3rdparty/nvshmem/build/src/lib/nvshmem_transport_ibrc.so.2 .
        cp -s ../../3rdparty/nvshmem/build/src/lib/libnvshmem_host.so.2 .
        export FLUX_SHM_USE_NVSHMEM=1
    fi
    popd
    ##### build flux torch bindings #####
    MAX_JOBS=${JOBS} python3 setup.py develop --user
    if [ $BDIST_WHEEL == "ON" ]; then
        MAX_JOBS=${JOBS} python3 setup.py bdist_wheel
    fi
    merge_compile_commands
}

NCCL_ROOT=$PROJECT_ROOT/3rdparty/nccl
build_nccl


if [ $ENABLE_NVSHMEM == "ON" ]; then
    ./build_nvshmem.sh ${build_args} --jobs ${JOBS}
fi

build_protobuf

if [ $ENABLE_NVSHMEM == "ON" ]; then
    build_pynvshmem
fi

build_flux_cuda
build_flux_py
