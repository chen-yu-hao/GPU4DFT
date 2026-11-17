#include "gpu4dft/multigpu.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime.h>
namespace gpu4dft::detail {
int cuda_visible_devices() noexcept;
void set_int_device(int* d_ptr, int value, StreamHandle stream);
std::vector<int> run_worksteal_cuda(int device_id, int* d_counter, int num_tasks,
                                    const int* h_weights, double* h_out, int num_blocks,
                                    StreamHandle stream);
void add_inplace_cuda(std::size_t count, const double* d_src, double* d_dst, StreamHandle stream);
void allreduce_multi_gpu_cuda(const std::vector<int>& device_ids,
                              const std::vector<StreamHandle>& streams,
                              const std::vector<std::unique_ptr<DeviceMemoryPool>>& pools,
                              const std::vector<std::vector<bool>>& p2p_matrix,
                              std::span<double* const> per_gpu_d_bufs, std::size_t count);
}  // namespace gpu4dft::detail
#endif

namespace gpu4dft {

// ---------------------------------------------------------------------------
// DynamicTaskQueue
// ---------------------------------------------------------------------------

struct DynamicTaskQueue::Impl {
  int device_id = 0;
  bool on_gpu = false;
  int* d_counter = nullptr;
  int host_counter = 0;

  explicit Impl(int dev) {
#if defined(GPU4DFT_HAVE_CUDA)
    const int nvis = detail::cuda_visible_devices();
    if (nvis > 0) {
      int prev_dev = 0;
      (void)cudaGetDevice(&prev_dev);
      device_id = (dev < 0) ? prev_dev : dev;
      if (device_id != prev_dev) (void)cudaSetDevice(device_id);
      if (cudaMalloc(&d_counter, sizeof(int)) == cudaSuccess) {
        on_gpu = true;
        detail::set_int_device(d_counter, 0, nullptr);
      } else {
        (void)cudaGetLastError();
      }
      if (device_id != prev_dev) (void)cudaSetDevice(prev_dev);
    }
#else
    (void)dev;
#endif
    if (!on_gpu) {
      d_counter = &host_counter;
    }
  }

  ~Impl() {
#if defined(GPU4DFT_HAVE_CUDA)
    if (on_gpu && d_counter != nullptr) {
      int prev_dev = 0;
      (void)cudaGetDevice(&prev_dev);
      if (device_id != prev_dev) (void)cudaSetDevice(device_id);
      (void)cudaFree(d_counter);
      if (device_id != prev_dev) (void)cudaSetDevice(prev_dev);
    }
#endif
  }
};

DynamicTaskQueue::DynamicTaskQueue(int device_id) : impl_(std::make_unique<Impl>(device_id)) {}
DynamicTaskQueue::~DynamicTaskQueue() = default;
DynamicTaskQueue::DynamicTaskQueue(DynamicTaskQueue&&) noexcept = default;
DynamicTaskQueue& DynamicTaskQueue::operator=(DynamicTaskQueue&&) noexcept = default;

void DynamicTaskQueue::reset(int initial_value, StreamHandle stream) {
#if defined(GPU4DFT_HAVE_CUDA)
  if (impl_->on_gpu) {
    int prev_dev = 0;
    (void)cudaGetDevice(&prev_dev);
    if (impl_->device_id != prev_dev) (void)cudaSetDevice(impl_->device_id);
    detail::set_int_device(impl_->d_counter, initial_value, stream);
    if (impl_->device_id != prev_dev) (void)cudaSetDevice(prev_dev);
    return;
  }
#else
  (void)stream;
#endif
  impl_->host_counter = initial_value;
}

int* DynamicTaskQueue::device_counter() noexcept { return impl_->d_counter; }
const int* DynamicTaskQueue::device_counter() const noexcept { return impl_->d_counter; }

int DynamicTaskQueue::value(StreamHandle stream) const {
#if defined(GPU4DFT_HAVE_CUDA)
  if (impl_->on_gpu) {
    int prev_dev = 0;
    (void)cudaGetDevice(&prev_dev);
    if (impl_->device_id != prev_dev) (void)cudaSetDevice(impl_->device_id);
    int h_val = 0;
    copy_d2h_async(&h_val, impl_->d_counter, sizeof(int), stream);
    synchronize_stream(stream);
    if (impl_->device_id != prev_dev) (void)cudaSetDevice(prev_dev);
    return h_val;
  }
#else
  (void)stream;
#endif
  return impl_->host_counter;
}

std::vector<int> DynamicTaskQueue::execute_weighted_tasks(int num_tasks, const int* h_weights,
                                                          double* h_out, int num_worker_blocks,
                                                          StreamHandle stream) {
  if (num_tasks <= 0) return {};
#if defined(GPU4DFT_HAVE_CUDA)
  if (impl_->on_gpu) {
    return detail::run_worksteal_cuda(impl_->device_id, impl_->d_counter, num_tasks, h_weights,
                                      h_out, num_worker_blocks, stream);
  }
#else
  (void)stream;
#endif
  num_worker_blocks = std::max(1, num_worker_blocks);
  impl_->host_counter = 0;
  std::vector<int> counts(static_cast<std::size_t>(num_worker_blocks), 0);
  int worker = 0;
  for (;;) {
    const int task = claim_next_task(&impl_->host_counter, 1);
    if (task >= num_tasks) break;
    counts[worker % num_worker_blocks] += 1;
    worker += 1;
    const int iters = std::max(1, h_weights[task]);
    double acc = 0.0;
    for (int k = 0; k < iters; ++k) {
      const double x = static_cast<double>(task + 1) * 1e-3 + static_cast<double>(k + 1) * 1e-5;
      acc += 1.0 / (1.0 + x * x);
    }
    h_out[task] = acc;
  }
  return counts;
}

// ---------------------------------------------------------------------------
// MultiGpuManager
// ---------------------------------------------------------------------------

struct MultiGpuManager::Impl {
  std::vector<int> device_ids;
  std::vector<StreamHandle> streams;
  std::vector<std::unique_ptr<DeviceMemoryPool>> pools;
  std::vector<std::vector<bool>> p2p_matrix;
  std::vector<double*> registered_peer_bufs;
  bool has_cuda = false;

  explicit Impl(std::vector<int> ids) {
#if defined(GPU4DFT_HAVE_CUDA)
    const int nvis = detail::cuda_visible_devices();
    if (nvis > 0) {
      has_cuda = true;
      if (ids.empty()) {
        ids.resize(static_cast<std::size_t>(nvis));
        std::iota(ids.begin(), ids.end(), 0);
      }
      for (int id : ids) {
        if (id < 0 || id >= nvis) {
          throw std::invalid_argument("MultiGpuManager: device_id out of range [0, visible)");
        }
      }
      device_ids = std::move(ids);
      const std::size_t ngpu = device_ids.size();
      streams.assign(ngpu, nullptr);
      pools.resize(ngpu);
      p2p_matrix.assign(ngpu, std::vector<bool>(ngpu, false));

      int prev_dev = 0;
      (void)cudaGetDevice(&prev_dev);
      for (std::size_t r = 0; r < ngpu; ++r) {
        cudaSetDevice(device_ids[r]);
        cudaStream_t s = nullptr;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
          (void)cudaGetLastError();
        }
        streams[r] = s;
        pools[r] = std::make_unique<DeviceMemoryPool>(device_ids[r]);
      }

      for (std::size_t a = 0; a < ngpu; ++a) {
        p2p_matrix[a][a] = true;
        cudaSetDevice(device_ids[a]);
        for (std::size_t b = 0; b < ngpu; ++b) {
          if (a == b) continue;
          int can_access = 0;
          if (cudaDeviceCanAccessPeer(&can_access, device_ids[a], device_ids[b]) == cudaSuccess &&
              can_access != 0) {
            cudaError_t peer_err = cudaDeviceEnablePeerAccess(device_ids[b], 0);
            if (peer_err == cudaSuccess || peer_err == cudaErrorPeerAccessAlreadyEnabled) {
              if (peer_err == cudaErrorPeerAccessAlreadyEnabled) (void)cudaGetLastError();
              p2p_matrix[a][b] = true;
            } else {
              (void)cudaGetLastError();
            }
          } else {
            (void)cudaGetLastError();
          }
        }
      }
      (void)cudaSetDevice(prev_dev);
      return;
    }
#else
    (void)ids;
#endif
    // Host-only / zero-GPU fallback slot
    device_ids = {0};
    streams = {nullptr};
    pools.push_back(std::make_unique<DeviceMemoryPool>(0));
    p2p_matrix = {{true}};
  }

  ~Impl() {
#if defined(GPU4DFT_HAVE_CUDA)
    if (has_cuda) {
      int prev_dev = 0;
      (void)cudaGetDevice(&prev_dev);
      for (std::size_t r = 0; r < device_ids.size(); ++r) {
        if (streams[r] != nullptr) {
          cudaSetDevice(device_ids[r]);
          (void)cudaStreamSynchronize(streams[r]);
          (void)cudaStreamDestroy(streams[r]);
          streams[r] = nullptr;
        }
      }
      (void)cudaSetDevice(prev_dev);
    }
#endif
  }
};

MultiGpuManager::MultiGpuManager(std::vector<int> device_ids)
    : impl_(std::make_unique<Impl>(std::move(device_ids))) {}

MultiGpuManager::~MultiGpuManager() = default;
MultiGpuManager::MultiGpuManager(MultiGpuManager&&) noexcept = default;
MultiGpuManager& MultiGpuManager::operator=(MultiGpuManager&&) noexcept = default;

int MultiGpuManager::visible_device_count() noexcept {
#if defined(GPU4DFT_HAVE_CUDA)
  return detail::cuda_visible_devices();
#else
  return 0;
#endif
}

int MultiGpuManager::num_devices() const noexcept {
  return impl_->has_cuda ? static_cast<int>(impl_->device_ids.size()) : 0;
}

int MultiGpuManager::device_id(int rank) const {
  return impl_->device_ids.at(static_cast<std::size_t>(rank));
}

StreamHandle MultiGpuManager::stream(int rank) const {
  return impl_->streams.at(static_cast<std::size_t>(rank));
}

DeviceMemoryPool& MultiGpuManager::pool(int rank) {
  return *impl_->pools.at(static_cast<std::size_t>(rank));
}

bool MultiGpuManager::p2p_enabled(int rank_a, int rank_b) const {
  return impl_->p2p_matrix.at(static_cast<std::size_t>(rank_a))
      .at(static_cast<std::size_t>(rank_b));
}

void MultiGpuManager::synchronize_all() {
#if defined(GPU4DFT_HAVE_CUDA)
  if (impl_->has_cuda) {
    int prev_dev = 0;
    (void)cudaGetDevice(&prev_dev);
    for (std::size_t r = 0; r < impl_->device_ids.size(); ++r) {
      cudaSetDevice(impl_->device_ids[r]);
      synchronize_stream(impl_->streams[r]);
    }
    (void)cudaSetDevice(prev_dev);
  }
#endif
}

void MultiGpuManager::register_peer_buffers(std::span<double* const> per_gpu_d_bufs) {
  impl_->registered_peer_bufs.assign(per_gpu_d_bufs.begin(), per_gpu_d_bufs.end());
}

void MultiGpuManager::allreduce_sum(double* d_buf, std::size_t count) {
  if (count == 0 || d_buf == nullptr) return;
  if (!impl_->registered_peer_bufs.empty() &&
      impl_->registered_peer_bufs.size() == impl_->device_ids.size() &&
      impl_->device_ids.size() > 1) {
    allreduce_sum(std::span<double* const>(impl_->registered_peer_bufs), count);
    return;
  }
  // Single-GPU communicator: sum of 1 rank is already in d_buf.
}

void MultiGpuManager::allreduce_sum(std::span<double* const> per_gpu_d_bufs, std::size_t count) {
  if (per_gpu_d_bufs.size() <= 1 || count == 0) return;
#if defined(GPU4DFT_HAVE_CUDA)
  if (impl_->has_cuda) {
    detail::allreduce_multi_gpu_cuda(impl_->device_ids, impl_->streams, impl_->pools,
                                     impl_->p2p_matrix, per_gpu_d_bufs, count);
    return;
  }
#endif
  for (std::size_t r = 1; r < per_gpu_d_bufs.size(); ++r) {
    for (std::size_t i = 0; i < count; ++i) {
      per_gpu_d_bufs[0][i] += per_gpu_d_bufs[r][i];
    }
  }
  for (std::size_t r = 1; r < per_gpu_d_bufs.size(); ++r) {
    std::copy(per_gpu_d_bufs[0], per_gpu_d_bufs[0] + count, per_gpu_d_bufs[r]);
  }
}

void MultiGpuManager::allreduce_local_partitions(std::span<double* const> d_partition_bufs,
                                                 std::size_t count, StreamHandle stream) {
  if (d_partition_bufs.size() <= 1 || count == 0) return;
#if defined(GPU4DFT_HAVE_CUDA)
  if (impl_->has_cuda) {
    for (std::size_t p = 1; p < d_partition_bufs.size(); ++p) {
      detail::add_inplace_cuda(count, d_partition_bufs[p], d_partition_bufs[0], stream);
    }
    for (std::size_t p = 1; p < d_partition_bufs.size(); ++p) {
      copy_d2d_async(d_partition_bufs[p], d_partition_bufs[0], count * sizeof(double), stream);
    }
    synchronize_stream(stream);
    return;
  }
#else
  (void)stream;
#endif
  for (std::size_t p = 1; p < d_partition_bufs.size(); ++p) {
    for (std::size_t i = 0; i < count; ++i) {
      d_partition_bufs[0][i] += d_partition_bufs[p][i];
    }
  }
  for (std::size_t p = 1; p < d_partition_bufs.size(); ++p) {
    std::copy(d_partition_bufs[0], d_partition_bufs[0] + count, d_partition_bufs[p]);
  }
}

}  // namespace gpu4dft
