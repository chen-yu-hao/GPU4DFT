#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "gpu4dft/config.hpp"
#include "gpu4dft/device_pool.hpp"

namespace gpu4dft {

// Device/host helper for persistent worker blocks claiming tiles from a dynamic
// atomic counter (`int tile = claim_next_task(d_counter, 1);`).
GPU4DFT_HD inline int claim_next_task(int* counter, int batch_size = 1) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return atomicAdd(counter, batch_size);
#else
  const int prev = *counter;
  *counter += batch_size;
  return prev;
#endif
}

// Device-resident atomic counter for dynamic load balancing across non-uniform
// shell-quartet or grid-batch tiles.
class DynamicTaskQueue {
 public:
  explicit DynamicTaskQueue(int device_id = -1);
  ~DynamicTaskQueue();

  DynamicTaskQueue(const DynamicTaskQueue&) = delete;
  DynamicTaskQueue& operator=(const DynamicTaskQueue&) = delete;
  DynamicTaskQueue(DynamicTaskQueue&&) noexcept;
  DynamicTaskQueue& operator=(DynamicTaskQueue&&) noexcept;

  void reset(int initial_value = 0, StreamHandle stream = nullptr);

  // Device pointer suitable for `atomicAdd(queue.device_counter(), batch)` inside GPU kernels.
  int* device_counter() noexcept;
  const int* device_counter() const noexcept;

  // Reads current counter value (synchronizes `stream`).
  int value(StreamHandle stream = nullptr) const;

  // Executes a persistent-block GPU dynamic work-stealing pass over `num_tasks` tasks
  // with heterogeneous iteration weights `h_weights[t]`, writing deterministic results
  // to `h_out[t]` and returning per-block claimed task counts for load-balance verification.
  std::vector<int> execute_weighted_tasks(int num_tasks, const int* h_weights, double* h_out,
                                          int num_worker_blocks = 32,
                                          StreamHandle stream = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Manages visible GPUs, per-device non-blocking CUDA streams, peer-to-peer (P2P)
// memory access, and NCCL / P2P AllReduce summation for distributed Fock builds.
class MultiGpuManager {
 public:
  // If `device_ids` is empty, initializes across all visible CUDA devices (or 1 host slot if none).
  explicit MultiGpuManager(std::vector<int> device_ids = {});
  ~MultiGpuManager();

  MultiGpuManager(const MultiGpuManager&) = delete;
  MultiGpuManager& operator=(const MultiGpuManager&) = delete;
  MultiGpuManager(MultiGpuManager&&) noexcept;
  MultiGpuManager& operator=(MultiGpuManager&&) noexcept;

  static int visible_device_count() noexcept;

  int num_devices() const noexcept;
  int device_id(int rank) const;
  StreamHandle stream(int rank) const;
  DeviceMemoryPool& pool(int rank);
  bool p2p_enabled(int rank_a, int rank_b) const;
  void synchronize_all();

  // Register per-GPU buffer pointers so single-argument `allreduce_sum(d_buf, count)`
  // can sum across all registered peer buffers.
  void register_peer_buffers(std::span<double* const> per_gpu_d_bufs);

  // Sum `count` doubles in-place across the active communicator (`d_buf` on rank 0;
  // instant no-op on single GPU when no peer buffers are registered).
  void allreduce_sum(double* d_buf, std::size_t count);

  // Sum `count` doubles in-place across `per_gpu_d_bufs` (`per_gpu_d_bufs[r]` resides on
  // `device_id(r)`), using NCCL when compiled with GPU4DFT_HAVE_NCCL or CUDA P2P / pinned staging.
  void allreduce_sum(std::span<double* const> per_gpu_d_bufs, std::size_t count);

  // Sum `count` doubles in-place across multiple device buffers on the current GPU
  // (e.g. multi-stream / multi-partition Fock buffers), writing the global sum back to every buffer.
  void allreduce_local_partitions(std::span<double* const> d_partition_bufs, std::size_t count,
                                  StreamHandle stream = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu4dft
