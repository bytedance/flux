//===- topo_utils.cc ---------------------------------------------- C++ ---===//
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

#include "flux/ths_op/topo_utils.h"
#include <mutex>
#include <numeric>
#include <ostream>
#include <vector>
#include <set>
#include <iostream>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
// private header of NCCL. use at control
#include <nccl.h>
#include <nccl/detail/include/comm.h>
#include <nccl/detail/include/graph.h>
#include <nccl/detail/graph/topo.h>  // needed by ncclTopoSystem

namespace bytedance::flux::topo_utils {

struct TopoInfo {
  bool is_properly_placed;
  int total_world_size;
  std::array<int, PATH_DIS> world_size;
};

std::ostream &
operator<<(std::ostream &os, const TopoInfo &topo_info) {
  os << "is_properly_placed: " << topo_info.is_properly_placed << "\n";
  os << "total_world_size: " << topo_info.total_world_size << "\n";
  os << "world_size: ";
  for (int i = 0; i < PATH_DIS; i++) {
    os << topoPathTypeStr[i] << " " << topo_info.world_size[i] << " ";
  }
  return os;
}

std::ostream &
operator<<(std::ostream &os, const ncclTopoSystem &system) {
  for (int i = 0; i < system.nodes[GPU].count; i++) {
    const auto &node_i = system.nodes[GPU].nodes[i];
    os << "GPU" << node_i.gpu.rank << ":\t";
    for (int j = 0; j < system.nodes[GPU].count; j++) {
      // const auto &node_j = system.nodes[GPU].nodes[j];
      os << node_i.paths[GPU][j].type << "\t";
    }
    os << "\n";
  }
  return os;
}

namespace {
static TopoInfo topo_info;
static bool initialized = false;
static std::mutex mutex;
}  // namespace

struct UnionFoundSet {
  UnionFoundSet(int size) : parent(size) { std::iota(parent.begin(), parent.end(), 0); }

  int
  find(int x) {  // path compress to root
    while (x != parent[x])
      x = parent[x];
    return x;
  }

  bool
  merge(int x, int y) {
    int root_x = find(x);
    int root_y = find(y);
    if (root_x == root_y) {  // already merged
      return false;
    }
    parent[root_x] = root_y;  // make root_x a child of root_y
    return true;
  }

  bool
  is_connected(int x, int y) {
    return find(x) == find(y);
  }

  void
  flatten() {
    for (int i = 0; i < (int)parent.size(); i++) {
      parent[i] = find(i);
    }
  }

  std::vector<int> parent;
};

void
initialize_topo_with_nccl_comm(ncclComm_t comm_, int rank = 0) {
  auto *comm = (ncclComm *)comm_;
  const auto &gpu_nodes = comm->topo->nodes[GPU];
  if (rank == 0) {
    LOG(INFO) << "topo system: \n" << *(comm->topo);
  }

  for (int link_type = 0; link_type < PATH_DIS; link_type++) {
    // LOG(INFO) << "link_type: " << topoPathTypeStr[link_type] << "\n";
    // merge into a Union-Find-Set
    UnionFoundSet ufs(gpu_nodes.count);
    for (int i = 0; i < gpu_nodes.count; i++) {
      const ncclTopoNode &node_i = gpu_nodes.nodes[i];
      for (int j = i + 1; j < gpu_nodes.count; j++) {  // suppose bidirection connected
        if (node_i.paths[GPU][j].type <= link_type) {
          // i,j are connected. set in union_found set
          ufs.merge(i, j);
        }
      }
    }
    ufs.flatten();

    std::vector<std::vector<int>> connected_ranks_list;
    std::set<int> roots(ufs.parent.begin(), ufs.parent.end());  // unique

    int world_size = -1;
    for (auto root : roots) {
      std::vector<int> connected_ranks, connected_nodes;
      for (int i = 0; i < gpu_nodes.count; i++) {
        if (ufs.parent[i] == root) {
          connected_nodes.push_back(i);
          connected_ranks.push_back(gpu_nodes.nodes[i].gpu.rank);
        }
      }
      if (world_size == -1) {
        world_size = connected_ranks.size();
      }
      if (world_size != (int)connected_ranks.size()) {
        if (link_type != PATH_PIX) {
          topo_info.is_properly_placed = false;
          return;
        }
      }
      std::sort(connected_ranks.begin(), connected_ranks.end());
      for (size_t j = 1; j < connected_ranks.size(); j++) {  // suppose they're neighbours
        if (connected_ranks[j] - connected_ranks[j - 1] != 1) {
          topo_info.is_properly_placed = false;
          return;
        }
      }
      // check if all connected to all
      connected_ranks_list.push_back(connected_ranks);
    }
    topo_info.world_size[link_type] = world_size;
  }
  topo_info.is_properly_placed = true;
}

ncclComm_t
create_nccl_comm_with_processgroup(c10d::ProcessGroup &pg) {
  //-- NCCL--
  ncclComm_t nccl_comm;
  int rank = pg.getRank();
  auto stream = c10::cuda::getCurrentCUDAStream();
  void *hptr_id = nullptr;
  CUDA_CHECK(cudaMallocHost(&hptr_id, sizeof(ncclUniqueId)));

  ncclUniqueId &id = *static_cast<ncclUniqueId *>(hptr_id);
  if (rank == 0) {
    ncclGetUniqueId(&id);
  }

  std::vector<at::Tensor> src_gpus{at::empty(
      {sizeof(ncclUniqueId)},
      at::TensorOptions()
          .dtype(at::ScalarType::Byte)
          .device(at::kCUDA)
          .device_index(c10::cuda::current_device()))};
  c10::cuda::memcpy_and_sync(
      src_gpus[0].data_ptr(), hptr_id, sizeof(ncclUniqueId), cudaMemcpyHostToDevice, stream);

  auto work = pg.broadcast(src_gpus);
  work->wait();
  c10::cuda::memcpy_and_sync(
      hptr_id, src_gpus[0].data_ptr(), sizeof(ncclUniqueId), cudaMemcpyDeviceToHost, stream);
  NCCL_CHECK(ncclCommInitRank(&nccl_comm, pg.getSize(), id, rank));
  CUDA_CHECK(cudaFreeHost(hptr_id));
  return nccl_comm;
}

bool
is_topo_initialized() {
  std::lock_guard<std::mutex> _(mutex);
  return initialized;
}

void
initialize_topo(c10d::ProcessGroup &group) {
  std::lock_guard<std::mutex> _(mutex);
  if (initialized) {
    std::cerr << "already initialized\n";
    return;
  }
  int rank = group.getRank();
  auto nccl_comm = create_nccl_comm_with_processgroup(group);
  topo_info.total_world_size = group.getSize();
  initialize_topo_with_nccl_comm(nccl_comm, rank);
  initialized = true;
  // create a ncclComm_t only to get the topo
  if (rank == 0) {
    LOG(INFO) << "topo_info:" << topo_info;
  }
  NCCL_CHECK(ncclCommDestroy(nccl_comm));
}

bool
is_topo_properly_placed() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.is_properly_placed;
}

bool
has_nvswitch() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.is_properly_placed &&
         topo_info.world_size[PATH_NVL] == topo_info.total_world_size;
}

bool
has_heterogeneous_nvlink() {  // has NVLink but no NVSwitch
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.is_properly_placed && topo_info.world_size[PATH_NVL] > 1 &&
         topo_info.world_size[PATH_NVL] != topo_info.total_world_size;
}

int
topo_nvlink_local_world_size() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.is_properly_placed ? topo_info.world_size[PATH_NVL] : -1;
}

bool
has_nvlink() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.is_properly_placed && topo_info.world_size[PATH_NVL] > 1;
}

bool
has_heterogeneous_pcie() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.is_properly_placed &&
         topo_info.world_size[PATH_PIX] != topo_info.total_world_size;
}

int
topo_numa_local_world_size() {
  std::lock_guard<std::mutex> _(mutex);
  FLUX_CHECK(initialized);
  return topo_info.is_properly_placed ? topo_info.world_size[PATH_PHB] : -1;
}

}  // namespace bytedance::flux::topo_utils
