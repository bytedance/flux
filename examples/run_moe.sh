#!/bin/bash

# This run script provides a minimal example of how to build MoE layers with Flux, 
# compared with the native pytorch implementation.

# The MoE layer0:
./launch.sh moe_layer0.py

# The MoE layer1:
./launch.sh moe_layer1.py

# For a complete and more detailed implementation of the MoE layer, please refer to docs/moe_usage.md