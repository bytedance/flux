#!/bin/bash
set -x
set -e

## Change export PATH if cuda is not at default path
export PATH=/usr/local/cuda/bin:$PATH
CMAKE=${CMAKE:-cmake}

ARCH=""
BUILD_TEST="ON"
BDIST_WHEEL="OFF"
WITH_PROTOBUF="OFF"
FLUX_DEBUG="OFF"
ENABLE_NVSHMEM="OFF"
WITH_TRITON_AOT="OFF"

function clean_py() {
    rm -rf build/lib.*
    rm -rf python/lib
    rm -rf .egg/
    rm -rf python/flux.egg-info
    rm -rf python/flux_ths_pybind.*
}

function clean_all() {
    clean_py
    rm -rf build/
    rm -rf 3rdparty/nvshmem/build
    rm -rf 3rdparty/nccl/build
    rm -rf 3rdparty/protobuf/build
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
    --triton-aot)
        WITH_TRITON_AOT="ON"
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
        CXXFLAGS_EXTRA=""
        use_cxx11_abi=$(python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")
        if [ $use_cxx11_abi == "False" ]; then
            CXXFLAGS_EXTRA="-D_GLIBCXX_USE_CXX11_ABI=0"
        fi
        CFLAGS="-fPIC" CXXFLAGS="-fPIC ${CXXFLAGS_EXTRA}" cmake ../cmake \
            -Dprotobuf_BUILD_TESTS=OFF \
            -Dprotobuf_BUILD_SHARED_LIBS=OFF \
            -DCMAKE_INSTALL_PREFIX=$(realpath local)
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
    cp -P -v ${BUILDDIR}/lib/lib* ${PREFIX}/lib/
    cp -P -v -r ${BUILDDIR}/include ${PREFIX}/
    popd
}

##### build flux_cuda #####
function build_flux_cuda() {
    mkdir -p build
    pushd build
    export LIBFLUX_PREFIX=${PROJECT_ROOT}/python/flux
    if [ ! -f CMakeCache.txt ] || [ -z ${FLUX_BUILD_SKIP_CMAKE} ]; then
        CMAKE_ARGS=(
            -DENABLE_NVSHMEM=${ENABLE_NVSHMEM}
            -DCUDAARCHS=${ARCH}
            -DCMAKE_EXPORT_COMPILE_COMMANDS=1
            -DBUILD_TEST=${BUILD_TEST}
            -DCMAKE_INSTALL_PREFIX=${LIBFLUX_PREFIX}
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
        if [ $WITH_TRITON_AOT == "ON" ]; then
            CMAKE_ARGS+=(
                -DWITH_TRITON_AOT=ON
            )
            export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/python
        fi
        ${CMAKE} .. ${CMAKE_ARGS[@]}
    fi
    make -j${JOBS} VERBOSE=1
    make install
    popd
}

function merge_compile_commands() {
    cd $SCRIPT_DIR
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
    LIBDIR=${PROJECT_ROOT}/python/flux/lib
    mkdir -p ${LIBDIR}

    pushd ${LIBDIR}
    if [ $ENABLE_NVSHMEM == "ON" ]; then
        cp -s -f ../../../3rdparty/nvshmem/build/src/lib/nvshmem_bootstrap_uid.so .
        cp -s -f ../../../3rdparty/nvshmem/build/src/lib/nvshmem_transport_ibrc.so.3 .
        cp -s -f ../../../3rdparty/nvshmem/build/src/lib/libnvshmem_host.so.3 .
        export FLUX_SHM_USE_NVSHMEM=1
    fi
    popd
    ##### build flux torch bindings #####
    MAX_JOBS=${JOBS} python3 setup.py develop --user
    if [ $BDIST_WHEEL == "ON" ]; then
        MAX_JOBS=${JOBS} python3 setup.py bdist_wheel
    fi
}

trap merge_compile_commands EXIT
NCCL_ROOT=$PROJECT_ROOT/3rdparty/nccl
build_nccl

if [ $ENABLE_NVSHMEM == "ON" ]; then
    ./build_nvshmem.sh ${build_args} --jobs ${JOBS}
fi

build_protobuf
build_flux_cuda
build_flux_py
