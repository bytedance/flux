#pragma once

#include "transfers.hpp"

#define FLUX_AG_RING_ORDER                                                                      \
  TransferCudaMemcpyPush<0, 1>, TransferCudaMemcpyPush<1, 2>, TransferCudaMemcpyPush<2, 3>,     \
      TransferCudaMemcpyPush<3, 4>, TransferCudaMemcpyPush<4, 5>, TransferCudaMemcpyPush<5, 6>, \
      TransferCudaMemcpyPush<6, 7>, TransferCudaMemcpyPush<7, 0>
// #define FLUX_AG_RING_ORDER TransferCudaMemcpyPull<0, 1>, TransferCudaMemcpyPull<1, 2>,
// TransferCudaMemcpyPull<2, 3>, TransferCudaMemcpyPull<3, 4>, TransferCudaMemcpyPull<4, 5>,
// TransferCudaMemcpyPull<5, 6>, TransferCudaMemcpyPull<6, 7>, TransferCudaMemcpyPull<7, 0>
