
#pragma once
#include "flux/ths_op/topo_utils.h"
#include <iostream>

namespace bytedance::flux {
// All2All for nvlink mode. for NVLINK machine, default is 0
// Ring1D for 1d-ring. for PCI-e machine without GPUs cross NUMA nodes use ring 1d
// Ring2D for 2d-ring. for PCI-e machine with GPUs cross NUMA nodes defaults to ring_2d
// RingCustom for custom ring. for defining arbitrary ring at compile time
enum class AGRingMode {
  All2All = 0,
  Ring1D = 1,
  Ring2D = 2,
  RingCustom = 3,
  Auto = -1,
};
static const int intra_numa_world_size = 4;

static AGRingMode
get_ring_mode(AGRingMode ring_mode) {
      return AGRingMode::All2All;
}

}  // namespace bytedance::flux
