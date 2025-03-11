# Guide for Comet (MLSys25 Artifact Evaluation)
This git repo (Flux) contains the components for the paper "Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts". The main code of Comet is located in the `src` directory. In detail, the implementation of MoE layer0 is in `src/moe_ag_scatter` and the implementation of MoE layer1 is in `src/moe_gather_rs`.


## Quick installation and test
Hardware requirements - A single server with 8 Nvidia GPUs (Hopper/Ada Lovelace/Ampere). We recommend to use H100/H800.
Software requirements - Please prepare as the following steps:

```bash

# Quick installation
conda create -n comet_ae python=3.11 -y
conda activate comet_ae
pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install byte-flux==1.1.1

# Quick test
git clone https://github.com/bytedance/flux.git && cd flux/examples
bash run_moe.sh
```

The successful running of the above command will prove the usability of the code.

## Measure the MoE layer and E2E model latency
Next, we provide a guide to measure the latency of MoE layer and E2E model using Comet with Megatron-LM, on a node with 8 GPUs.

### Prepare the environment
```bash
# Under your workspace
# Install some basic dependencies

pip3 install flash-attn --no-build-isolation
pip3 install transformer_engine[pytorch]
git clone https://github.com/NVIDIA/apex
pushd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
popd
pip install git+https://github.com/fanshiqing/grouped_gemm@v1.0
pip install regex six pyyaml

# Megatron-LM
git clone https://github.com/ZSL98/Megatron-LM.git
# fastmoe (baseline1)
git clone https://github.com/ZSL98/fastmoe.git && cd fastmoe && python setup.py install && pip install dm-tree && cd ..
# tutel (baseline2)
git clone https://github.com/ZSL98/tutel && cd tutel && python setup.py install && cd ..

```

### Run the tests

```bash
cd Megatron-LM
bash ./grid_test.sh # Record the single MoE layer results to timing_results.csv
bash ./e2e_grid_test.sh # Record the e2e model results to e2e_timing_results.csv
```
You can modify the parameters such as `NUM_TOKENS` and `EXPERT_NUM` to see the results under different configurations. The scripts' output can be found in `Megatron-LM/timing_results.csv` and `Megatron-LM/e2e_timing_results.csv`. The feasibility of the scripts has been tested on both 8 L20 GPUs and 8 H100 GPUs.
