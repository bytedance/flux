//===- topo_utls.h ------------------------------------------------------ C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nccl.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace bytedance::flux::topo_utils {

ncclComm_t create_nccl_comm_with_processgroup(c10::intrusive_ptr<c10d::ProcessGroup> pg);

bool is_topo_initialized();
/**
 * call this function multi times, you will got some warnings and only runs once really
 * @param group: this should be a local group. if not, split it to a local group from outside
 */
void initialize_topo(c10::intrusive_ptr<c10d::ProcessGroup> group);
void initialize_topo(const std::vector<int> &device_ids);

// has any NV-link supported GPU exists
bool has_nvlink();
// has NV-Switch(means all GPUS are connected to each other by NV-Link)
bool has_nvswitch();
// has NVLink but not all-to-all connected. such as A100 PCI-e version with NVLink
bool has_heterogeneous_nvlink();
// nvlink world_size. if with NV-Switch, this equals local world_size. otherwise, return the P2P
// connected cluster size
int topo_nvlink_local_world_size();

// has PIC-e but not under the same NUMA node
bool has_heterogeneous_pcie();
// the Gpus under the same NUMA node
// NOTE: same GPUs under different NUMA nodes are expected
int topo_numa_local_world_size();
}  // namespace bytedance::flux::topo_utils
