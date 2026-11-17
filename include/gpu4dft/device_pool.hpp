#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace gpu4dft {

#if defined(GPU4DFT_HAVE_CUDA)
using StreamHandle = cudaStream_t;
#else
using StreamHandle = void*;
#endif

struct PoolStats {
  std::size_t allocated_bytes = 0;
  std::size_t in_use_bytes = 0;
  std::size_t cached_free_bytes = 0;
  std::size_t peak_in_use_bytes = 0;
  std::size_t arena_capacity_bytes = 0;
  std::size_t arena_used_bytes = 0;
  std::size_t cache_hits = 0;
  std::size_t async_allocations = 0;
};

// Stream-ordered GPU memory pool wrapping cudaMallocAsync/cudaFreeAsync plus
// deterministic LIFO buffer reuse and an optional pre-reserved scratch arena.
// Ensures zero cudaMalloc synchronizations inside iterative GPU SCF loops and
// preserves pointer stability across CUDA Graph captures and replays.
class DeviceMemoryPool {
 public:
  static constexpr std::size_t kAlignment = 256;

  explicit DeviceMemoryPool(int device_id = -1);
  ~DeviceMemoryPool();

  DeviceMemoryPool(const DeviceMemoryPool&) = delete;
  DeviceMemoryPool& operator=(const DeviceMemoryPool&) = delete;
  DeviceMemoryPool(DeviceMemoryPool&&) noexcept;
  DeviceMemoryPool& operator=(DeviceMemoryPool&&) noexcept;

  // Acquire a 256-byte-aligned buffer of at least `bytes`. Reuses cached blocks
  // in LIFO order with zero CUDA API overhead when available; otherwise uses
  // stream-ordered cudaMallocAsync (outside graph capture) or the scratch arena.
  void* acquire(std::size_t bytes, StreamHandle stream = nullptr);

  // Return a buffer acquired via `acquire` back to the pool cache without host
  // synchronization.
  void release(void* ptr, StreamHandle stream = nullptr) noexcept;

  template <typename T>
  T* acquire_typed(std::size_t count, StreamHandle stream = nullptr) {
    if (count == 0) return nullptr;
    return static_cast<T*>(acquire(count * sizeof(T), stream));
  }

  // Pre-allocate a contiguous scratch arena slab for O(1) bump allocation.
  void reserve_arena(std::size_t bytes, StreamHandle stream = nullptr);
  void* alloc_arena(std::size_t bytes);
  void reset_arena() noexcept;

  // Release all cached free blocks back to the CUDA stream-ordered allocator.
  void trim(StreamHandle stream = nullptr);

  int device_id() const noexcept;
  PoolStats stats() const;

  static std::size_t align_bytes(std::size_t bytes) noexcept {
    return (bytes + kAlignment - 1) & ~(kAlignment - 1);
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// RAII handle for a typed buffer leased from DeviceMemoryPool.
template <typename T>
class PooledBuffer {
 public:
  PooledBuffer() = default;
  PooledBuffer(DeviceMemoryPool& pool, std::size_t count, StreamHandle stream = nullptr)
      : pool_(&pool), stream_(stream), count_(count) {
    if (count_ > 0) {
      ptr_ = pool_->acquire_typed<T>(count_, stream_);
    }
  }

  ~PooledBuffer() { reset(); }

  PooledBuffer(const PooledBuffer&) = delete;
  PooledBuffer& operator=(const PooledBuffer&) = delete;

  PooledBuffer(PooledBuffer&& other) noexcept
      : pool_(other.pool_), ptr_(other.ptr_), stream_(other.stream_), count_(other.count_) {
    other.pool_ = nullptr;
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  PooledBuffer& operator=(PooledBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      pool_ = other.pool_;
      ptr_ = other.ptr_;
      stream_ = other.stream_;
      count_ = other.count_;
      other.pool_ = nullptr;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void reset() noexcept {
    if (pool_ != nullptr && ptr_ != nullptr) {
      pool_->release(ptr_, stream_);
    }
    ptr_ = nullptr;
    count_ = 0;
  }

  T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return count_; }
  bool empty() const noexcept { return count_ == 0; }

 private:
  DeviceMemoryPool* pool_ = nullptr;
  T* ptr_ = nullptr;
  StreamHandle stream_ = nullptr;
  std::size_t count_ = 0;
};

// Returns a process-wide default DeviceMemoryPool for the active (or specified) CUDA device.
DeviceMemoryPool& default_device_pool(int device_id = -1);

// Stream-ordered memory transfer & synchronization utilities.
void copy_h2d_async(void* dst_device, const void* src_host, std::size_t bytes,
                    StreamHandle stream = nullptr);
void copy_d2h_async(void* dst_host, const void* src_device, std::size_t bytes,
                    StreamHandle stream = nullptr);
void copy_d2d_async(void* dst_device, const void* src_device, std::size_t bytes,
                    StreamHandle stream = nullptr);
void memset_zero_async(void* dst_device, std::size_t bytes, StreamHandle stream = nullptr);
void synchronize_stream(StreamHandle stream = nullptr);

// Captures and replays fixed-topology GPU kernel/GEMM sequences on a dedicated
// non-blocking CUDA stream using CUDA Graphs, with automatic warmup and clean
// stream fallback if an operation does not support stream capture.
class CudaGraphExecutor {
 public:
  explicit CudaGraphExecutor(bool warmup_before_capture = true);
  ~CudaGraphExecutor();

  CudaGraphExecutor(const CudaGraphExecutor&) = delete;
  CudaGraphExecutor& operator=(const CudaGraphExecutor&) = delete;
  CudaGraphExecutor(CudaGraphExecutor&&) noexcept;
  CudaGraphExecutor& operator=(CudaGraphExecutor&&) noexcept;

  // Execute a fixed-topology GPU sequence identified by `graph_key`.
  // - If already captured, replays via `cudaGraphLaunch` with near-zero launch overhead.
  // - If `warmup_before_capture` is enabled, runs once on stream to warm up pool/cuBLAS
  //   state, captures on the second invocation, and replays on all subsequent invocations.
  // - Automatically falls back to direct stream execution if capture is unsupported.
  void execute(std::uint64_t graph_key, const std::function<void(StreamHandle)>& record_fn);

  // Explicitly capture `record_fn` into `graph_key` without executing it yet.
  // Returns true if capture succeeded, false if capture fell back.
  bool capture(std::uint64_t graph_key, const std::function<void(StreamHandle)>& record_fn);

  // Launch an explicitly captured graph; throws if `graph_key` is not captured.
  void launch(std::uint64_t graph_key);

  bool is_captured(std::uint64_t graph_key) const;
  void invalidate(std::uint64_t graph_key);
  void clear();
  void synchronize();

  StreamHandle stream() const noexcept;
  std::size_t capture_count() const noexcept;
  std::size_t replay_count() const noexcept;
  std::size_t fallback_count() const noexcept;

  // Helper to combine integer parameters and pointers into a 64-bit graph key.
  static std::uint64_t hash_combine(std::uint64_t seed, std::uint64_t value) noexcept {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gpu4dft
