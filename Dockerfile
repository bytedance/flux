FROM nvcr.io/nvidia/pytorch:24.07-py3

WORKDIR /workspace/flux

COPY . .

RUN pip install ninja packaging
RUN git submodule update --init --recursive
RUN bash ./install_deps.sh
RUN OMP_NUM_THREADS=128 ./build.sh --arch "80;89;90" --nvshmem