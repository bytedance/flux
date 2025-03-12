#!/bin/bash

# This run script provides a minimal example of how to build MoE layers with Flux, 
# compared with the native pytorch implementation.

# Suppress NCCL debugging info
export NCCL_DEBUG=WARN

# The MoE layer0:
../launch.sh moe_layer0.py

# The MoE layer1:
../launch.sh moe_layer1.py

# A minimal MoE layer - Compare torch and Flux:
# ../launch.sh moe.py

# A minimal MoE layer with only flux:
# ../launch.sh moe_flux_only.py

# For a complete and more detailed implementation of the MoE layer, please refer to docs/moe_usage.md