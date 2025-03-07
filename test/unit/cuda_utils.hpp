//===- cuda_utils.hpp --------------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#ifndef FLUX_TEST_CUDA_UTILS_H_
#define FLUX_TEST_CUDA_UTILS_H_
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <utility>
#include <numa.h>
#include "flux/cuda/cuda_common.h"

class ScopedDevice {
 public:
  ScopedDevice(int device) {
    CUDA_CHECK(cudaGetDevice(&device_id_old_));
    if (device_id_old_ != device) {
      CUDA_CHECK(cudaSetDevice(device));
    } else {
      device_id_old_ = -1;
    }
  }

  ~ScopedDevice() {  // set back device_id
    if (device_id_old_ >= 0) {
      CUDA_CHECK(cudaSetDevice(device_id_old_));
    }
  }

 private:
  int device_id_old_ = -1;  // -1 for do nothing in rollback
};

template <typename T>
class Vector {
 public:
  [[nodiscard]] T *
  ptr() const {
    return (T *)ptr_;
  }

  virtual ~Vector() {}

  operator T *() const { return ptr(); }
  virtual std::vector<T> cpu() = 0;

 protected:
  void *ptr_ = nullptr;
  std::shared_ptr<char> deleter_;
};

template <typename T>
class DeviceVector : public Vector<T> {
 public:
  DeviceVector() = default;
  DeviceVector(const DeviceVector &) = default;
  DeviceVector(int device_id, size_t size, int value) : size_(size * sizeof(T)), host_vec_(size) {
    ScopedDevice _(device_id);
    CUDA_CHECK(cudaMalloc(&ptr_, size_));
    CUDA_CHECK(cudaMemset(ptr_, value, size_));
    deleter_ = std::shared_ptr<char>(nullptr, [this](void *) { reset(); });
  }
  DeviceVector(DeviceVector &&other) noexcept {
    ptr_ = std::exchange(other.ptr_, nullptr);
    size_ = std::exchange(other.size_, 0);
    deleter_ = std::move(other.deleter_);
  }

  void
  reset() {
    if (ptr_) {
      CUDA_CHECK(cudaFree(ptr_));
    }
  }

  void
  sync_device() {
    CUDA_CHECK(cudaMemcpy(ptr_, host_vec_.data(), size_, cudaMemcpyHostToDevice));
  }

  void
  sync_host() {
    CUDA_CHECK(cudaMemcpy(host_vec_.data(), ptr_, size_, cudaMemcpyDeviceToHost));
  }

  std::vector<T>
  cpu() override {
    sync_host();
    return host_vec_;
  }

  T *
  cpu_ptr() {
    return host_vec_.data();
  }

 protected:
  using Vector<T>::ptr_;
  using Vector<T>::deleter_;

 private:
  size_t size_ = 0;
  std::vector<T> host_vec_;
};

template <typename T>
class PinHostVector : public Vector<T> {
 public:
  PinHostVector() = default;
  PinHostVector(const PinHostVector &) = default;
  PinHostVector(PinHostVector &&) = default;
  PinHostVector(size_t size) : size_(size * sizeof(T)) {
    CUDA_CHECK(cudaHostAlloc(&ptr_, size_, cudaHostAllocDefault));
    memset(ptr_, 0, size_);
    deleter_ = std::shared_ptr<char>(nullptr, [this](void *) { reset(); });
  }
  void
  reset() {
    if (ptr_) {
      CUDA_CHECK(cudaFreeHost(ptr_));
    }
  }

  std::vector<T>
  cpu() override {
    return std::vector<T>((T *)ptr_, (T *)ptr_ + size_ / sizeof(T));
  }

 protected:
  using Vector<T>::ptr_, Vector<T>::deleter_;

 private:
  size_t size_ = 0;
};

template <typename T>
class NumaVector : public Vector<T> {
 public:
  NumaVector() = default;
  NumaVector(const NumaVector &) = default;
  NumaVector(NumaVector &&) = default;
  // node < 0 for non-numa node
  NumaVector(size_t elems, int numa_node, bool pin_data = true)
      : size_(elems * sizeof(T)), numa_node_(numa_node), pin_data_(pin_data) {
    if (numa_node_ < 0) {
      ptr_ = new T[elems];
    } else {
      ptr_ = numa_alloc_onnode(size_, numa_node_);
    }
    FLUX_CHECK(ptr_ != nullptr);
    if (pin_data_) {
      CUDA_CHECK(cudaHostRegister(ptr_, size_, cudaHostRegisterDefault));
    }
    memset(ptr_, 0, size_);
    deleter_ = std::shared_ptr<char>(nullptr, [this](void *) { reset(); });
  }
  void
  reset() {
    if (ptr_) {
      if (pin_data_) {
        CUDA_CHECK(cudaHostUnregister(ptr_));
      }
      if (numa_node_ < 0) {
        delete[] (T *)ptr_;
      } else {
        numa_free(ptr_, size_);
      }
    }
  }

  std::vector<T>
  cpu() override {
    return std::vector<T>((T *)ptr_, (T *)ptr_ + size_ / sizeof(T));
  }

 protected:
  using Vector<T>::ptr_, Vector<T>::deleter_;

 private:
  size_t size_;
  int numa_node_;
  bool pin_data_;
};

class CudaStream {
 public:
  CudaStream(const CudaStream &) = default;
  CudaStream(CudaStream &&) = default;
  CudaStream(int device_id)
      : stream_(std::shared_ptr<cudaStream_t>(new cudaStream_t, [](cudaStream_t *s) {
          if (s != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(*s));
            delete s;
          }
        })) {
    ScopedDevice _(device_id);
    CUDA_CHECK(cudaStreamCreateWithFlags(&*stream_, cudaStreamNonBlocking));
  }

  ~CudaStream() {}

  operator cudaStream_t() const { return *stream_; }

  void
  sync() {
    CUDA_CHECK(cudaStreamSynchronize(*stream_));
  }

 private:
  std::shared_ptr<cudaStream_t> stream_;
};

class CudaEvent {
 public:
  CudaEvent(const CudaEvent &) = default;
  CudaEvent(CudaEvent &&) = default;
  CudaEvent(int device_id)
      : devid_(device_id), event_(new cudaEvent_t, [](cudaEvent_t *e) {
          if (e) {
            CUDA_CHECK(cudaEventDestroy(*e));
            delete e;
          }
        }) {
    ScopedDevice _(device_id);
    CUDA_CHECK(cudaEventCreate(&*event_));
  }

  ~CudaEvent() {}

  operator cudaEvent_t() const { return *event_; }

  void
  record(const cudaStream_t &stream) {
    ScopedDevice _(devid_);
    CUDA_CHECK(cudaEventRecord(*event_, stream));
  }

 private:
  int devid_;
  std::shared_ptr<cudaEvent_t> event_;
};

class CudaEventTimer {
 public:
  CudaEventTimer(int device_id)
      : start_event_(device_id), stop_event_(device_id), stream_(nullptr) {}

  void
  Start(cudaStream_t stream) {
    stream_ = stream;
    start_event_.record(stream);
  }

  void
  Stop() {
    stop_event_.record(stream_);
  }

  float
  GetEclapsedTime() {
    float duration;
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    CUDA_CHECK(cudaEventElapsedTime(&duration, start_event_, stop_event_));
    return duration;
  }

 private:
  CudaEvent start_event_;
  CudaEvent stop_event_;
  cudaStream_t stream_;
};
#endif
