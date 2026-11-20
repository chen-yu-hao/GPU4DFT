#include "gpu4dft/device_pool.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu4dft {

namespace {

inline void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

class ScopedDeviceGuard {
 public:
  explicit ScopedDeviceGuard(int target_device) {
    if (target_device >= 0) {
      if (cudaGetDevice(&prev_device_) == cudaSuccess && prev_device_ != target_device) {
        check_cuda(cudaSetDevice(target_device), "cudaSetDevice");
        switched_ = true;
      }
    }
  }
  ~ScopedDeviceGuard() {
    if (switched_) {
      (void)cudaSetDevice(prev_device_);
    }
  }
  ScopedDeviceGuard(const ScopedDeviceGuard&) = delete;
  ScopedDeviceGuard& operator=(const ScopedDeviceGuard&) = delete;

 private:
  int prev_device_ = 0;
  bool switched_ = false;
};

}  // namespace

struct DeviceMemoryPool::Impl {
  int device_id = 0;
  bool async_supported = false;

  mutable std::mutex mtx;
  std::map<std::size_t, std::vector<void*>> free_lists;
  std::unordered_map<void*, std::size_t> active_blocks;

  std::uint8_t* arena_base = nullptr;
  std::size_t arena_capacity = 0;
  std::size_t arena_offset = 0;

  std::size_t allocated_bytes = 0;
  std::size_t in_use_bytes = 0;
  std::size_t cached_free_bytes = 0;
  std::size_t peak_in_use_bytes = 0;
  std::size_t cache_hits = 0;
  std::size_t async_allocations = 0;

  explicit Impl(int dev) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
      (void)cudaGetLastError();
      device_id = 0;
      async_supported = false;
      return;
    }
    if (dev < 0) {
      check_cuda(cudaGetDevice(&device_id), "DeviceMemoryPool cudaGetDevice");
    } else {
      device_id = dev;
    }
    ScopedDeviceGuard guard(device_id);
    int attr = 0;
    if (cudaDeviceGetAttribute(&attr, cudaDevAttrMemoryPoolsSupported, device_id) == cudaSuccess &&
        attr != 0) {
      async_supported = true;
      cudaMemPool_t mempool{};
      if (cudaDeviceGetDefaultMemPool(&mempool, device_id) == cudaSuccess) {
        std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
        (void)cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
      } else {
        (void)cudaGetLastError();
      }
    } else {
      (void)cudaGetLastError();
      async_supported = false;
    }
  }

  bool pointer_in_arena(const void* ptr) const noexcept {
    if (arena_base == nullptr || ptr == nullptr) return false;
    const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    const auto base = reinterpret_cast<std::uintptr_t>(arena_base);
    return addr >= base && addr < (base + arena_capacity);
  }

  void free_raw_unlocked(void* ptr, StreamHandle stream) noexcept {
    if (ptr == nullptr) return;
    ScopedDeviceGuard guard(device_id);
    if (async_supported) {
      cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
      if (stream != nullptr && cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess &&
          cap_status != cudaStreamCaptureStatusNone) {
        (void)cudaFree(ptr);
      } else {
        if (cudaFreeAsync(ptr, stream) != cudaSuccess) {
          (void)cudaGetLastError();
          (void)cudaFree(ptr);
        }
      }
    } else {
      (void)cudaFree(ptr);
    }
  }

  ~Impl() {
    ScopedDeviceGuard guard(device_id);
    for (auto& kv : free_lists) {
      for (void* ptr : kv.second) {
        if (async_supported) {
          if (cudaFreeAsync(ptr, nullptr) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaFree(ptr);
          }
        } else {
          (void)cudaFree(ptr);
        }
      }
    }
    for (auto& kv : active_blocks) {
      if (async_supported) {
        if (cudaFreeAsync(kv.first, nullptr) != cudaSuccess) {
          (void)cudaGetLastError();
          (void)cudaFree(kv.first);
        }
      } else {
        (void)cudaFree(kv.first);
      }
    }
    if (arena_base != nullptr) {
      if (async_supported) {
        if (cudaFreeAsync(arena_base, nullptr) != cudaSuccess) {
          (void)cudaGetLastError();
          (void)cudaFree(arena_base);
        }
      } else {
        (void)cudaFree(arena_base);
      }
      arena_base = nullptr;
    }
  }
};

DeviceMemoryPool::DeviceMemoryPool(int device_id) : impl_(std::make_unique<Impl>(device_id)) {}
DeviceMemoryPool::~DeviceMemoryPool() = default;
DeviceMemoryPool::DeviceMemoryPool(DeviceMemoryPool&&) noexcept = default;
DeviceMemoryPool& DeviceMemoryPool::operator=(DeviceMemoryPool&&) noexcept = default;

void* DeviceMemoryPool::acquire(std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return nullptr;
  const std::size_t aligned = align_bytes(bytes);

  std::lock_guard<std::mutex> lock(impl_->mtx);
  auto it = impl_->free_lists.find(aligned);
  if (it != impl_->free_lists.end() && !it->second.empty()) {
    void* ptr = it->second.back();
    it->second.pop_back();
    if (it->second.empty()) {
      impl_->free_lists.erase(it);
    }
    impl_->active_blocks.emplace(ptr, aligned);
    impl_->cached_free_bytes -= aligned;
    impl_->in_use_bytes += aligned;
    impl_->peak_in_use_bytes = std::max(impl_->peak_in_use_bytes, impl_->in_use_bytes);
    impl_->cache_hits += 1;
    return ptr;
  }

  ScopedDeviceGuard guard(impl_->device_id);
  cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
  const bool capturing =
      (stream != nullptr && cudaStreamIsCapturing(stream, &cap_status) == cudaSuccess &&
       cap_status != cudaStreamCaptureStatusNone);
  if (capturing) {
    if (impl_->arena_base != nullptr && impl_->arena_offset + aligned <= impl_->arena_capacity) {
      void* ptr = impl_->arena_base + impl_->arena_offset;
      impl_->arena_offset += aligned;
      return ptr;
    }
    throw std::runtime_error(
        "DeviceMemoryPool::acquire called during CUDA Graph capture without pre-warmed cache or "
        "arena capacity");
  }

  // Size-tolerant reuse outside graph capture: reuse the smallest cached block >= aligned within
  // bounded slack so varying chunk dimensions (e.g. RI aux chunks in OzakiInt8) do not fragment.
  auto lb = impl_->free_lists.lower_bound(aligned);
  if (lb != impl_->free_lists.end() && !lb->second.empty() &&
      lb->first <= aligned * 2 && (lb->first - aligned) <= (256ULL << 20)) {
    const std::size_t block_bytes = lb->first;
    void* ptr = lb->second.back();
    lb->second.pop_back();
    if (lb->second.empty()) {
      impl_->free_lists.erase(lb);
    }
    impl_->active_blocks.emplace(ptr, block_bytes);
    if (impl_->cached_free_bytes >= block_bytes) {
      impl_->cached_free_bytes -= block_bytes;
    } else {
      impl_->cached_free_bytes = 0;
    }
    impl_->in_use_bytes += block_bytes;
    impl_->peak_in_use_bytes = std::max(impl_->peak_in_use_bytes, impl_->in_use_bytes);
    impl_->cache_hits += 1;
    return ptr;
  }

  void* ptr = nullptr;
  if (impl_->async_supported) {
    cudaError_t st = cudaMallocAsync(&ptr, aligned, stream);
    if (st == cudaErrorMemoryAllocation) {
      (void)cudaGetLastError();
      for (auto& kv : impl_->free_lists) {
        const std::size_t block_bytes = kv.first;
        for (void* fptr : kv.second) {
          impl_->free_raw_unlocked(fptr, stream);
          if (impl_->allocated_bytes >= block_bytes) {
            impl_->allocated_bytes -= block_bytes;
          }
        }
      }
      impl_->free_lists.clear();
      impl_->cached_free_bytes = 0;
      if (stream != nullptr) {
        (void)cudaStreamSynchronize(stream);
      }
      (void)cudaDeviceSynchronize();
      cudaMemPool_t mempool{};
      if (cudaDeviceGetDefaultMemPool(&mempool, impl_->device_id) == cudaSuccess) {
        (void)cudaMemPoolTrimTo(mempool, 0);
      } else {
        (void)cudaGetLastError();
      }
      st = cudaMallocAsync(&ptr, aligned, stream);
    }
    check_cuda(st, "DeviceMemoryPool cudaMallocAsync");
  } else {
    check_cuda(cudaMalloc(&ptr, aligned), "DeviceMemoryPool cudaMalloc");
  }

  impl_->active_blocks.emplace(ptr, aligned);
  impl_->allocated_bytes += aligned;
  impl_->in_use_bytes += aligned;
  impl_->peak_in_use_bytes = std::max(impl_->peak_in_use_bytes, impl_->in_use_bytes);
  impl_->async_allocations += 1;
  return ptr;
}

void DeviceMemoryPool::release(void* ptr, StreamHandle /*stream*/) noexcept {
  if (ptr == nullptr || !impl_) return;
  std::lock_guard<std::mutex> lock(impl_->mtx);
  if (impl_->pointer_in_arena(ptr)) {
    return;
  }
  auto it = impl_->active_blocks.find(ptr);
  if (it == impl_->active_blocks.end()) {
    return;
  }
  const std::size_t aligned = it->second;
  impl_->active_blocks.erase(it);
  impl_->free_lists[aligned].push_back(ptr);
  if (impl_->in_use_bytes >= aligned) {
    impl_->in_use_bytes -= aligned;
  }
  impl_->cached_free_bytes += aligned;
}

void DeviceMemoryPool::reserve_arena(std::size_t bytes, StreamHandle stream) {
  const std::size_t aligned = align_bytes(bytes);
  std::lock_guard<std::mutex> lock(impl_->mtx);
  if (aligned <= impl_->arena_capacity) {
    return;
  }
  ScopedDeviceGuard guard(impl_->device_id);
  if (impl_->arena_base != nullptr) {
    impl_->free_raw_unlocked(impl_->arena_base, stream);
    impl_->arena_base = nullptr;
    impl_->arena_capacity = 0;
    impl_->arena_offset = 0;
  }
  void* ptr = nullptr;
  if (impl_->async_supported) {
    check_cuda(cudaMallocAsync(&ptr, aligned, stream), "DeviceMemoryPool::reserve_arena");
  } else {
    check_cuda(cudaMalloc(&ptr, aligned), "DeviceMemoryPool::reserve_arena");
  }
  impl_->arena_base = static_cast<std::uint8_t*>(ptr);
  impl_->arena_capacity = aligned;
  impl_->arena_offset = 0;
}

void* DeviceMemoryPool::alloc_arena(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  const std::size_t aligned = align_bytes(bytes);
  std::lock_guard<std::mutex> lock(impl_->mtx);
  if (impl_->arena_base == nullptr || impl_->arena_offset + aligned > impl_->arena_capacity) {
    throw std::runtime_error("DeviceMemoryPool::alloc_arena exceeded reserved arena capacity");
  }
  void* ptr = impl_->arena_base + impl_->arena_offset;
  impl_->arena_offset += aligned;
  return ptr;
}

void DeviceMemoryPool::reset_arena() noexcept {
  if (!impl_) return;
  std::lock_guard<std::mutex> lock(impl_->mtx);
  impl_->arena_offset = 0;
}

void DeviceMemoryPool::trim(StreamHandle stream) {
  std::lock_guard<std::mutex> lock(impl_->mtx);
  ScopedDeviceGuard guard(impl_->device_id);
  for (auto& kv : impl_->free_lists) {
    const std::size_t block_bytes = kv.first;
    for (void* ptr : kv.second) {
      impl_->free_raw_unlocked(ptr, stream);
      if (impl_->allocated_bytes >= block_bytes) {
        impl_->allocated_bytes -= block_bytes;
      }
    }
  }
  impl_->free_lists.clear();
  impl_->cached_free_bytes = 0;
  if (impl_->async_supported) {
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
    }
    (void)cudaDeviceSynchronize();
    cudaMemPool_t mempool{};
    if (cudaDeviceGetDefaultMemPool(&mempool, impl_->device_id) == cudaSuccess) {
      (void)cudaMemPoolTrimTo(mempool, 0);
    } else {
      (void)cudaGetLastError();
    }
  }
}

int DeviceMemoryPool::device_id() const noexcept { return impl_ ? impl_->device_id : 0; }

PoolStats DeviceMemoryPool::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mtx);
  return PoolStats{
      impl_->allocated_bytes,
      impl_->in_use_bytes,
      impl_->cached_free_bytes,
      impl_->peak_in_use_bytes,
      impl_->arena_capacity,
      impl_->arena_offset,
      impl_->cache_hits,
      impl_->async_allocations,
  };
}

DeviceMemoryPool& default_device_pool(int device_id) {
  static std::mutex registry_mtx;
  static std::unordered_map<int, std::unique_ptr<DeviceMemoryPool>> pools;
  int target = device_id;
  if (target < 0) {
    if (cudaGetDevice(&target) != cudaSuccess) {
      (void)cudaGetLastError();
      target = 0;
    }
  }
  std::lock_guard<std::mutex> lock(registry_mtx);
  auto& slot = pools[target];
  if (!slot) {
    slot = std::make_unique<DeviceMemoryPool>(target);
  }
  return *slot;
}

void copy_h2d_async(void* dst_device, const void* src_host, std::size_t bytes,
                    StreamHandle stream) {
  if (bytes == 0) return;
  check_cuda(cudaMemcpyAsync(dst_device, src_host, bytes, cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync H2D");
}

void copy_d2h_async(void* dst_host, const void* src_device, std::size_t bytes,
                    StreamHandle stream) {
  if (bytes == 0) return;
  check_cuda(cudaMemcpyAsync(dst_host, src_device, bytes, cudaMemcpyDeviceToHost, stream),
             "cudaMemcpyAsync D2H");
}

void copy_d2d_async(void* dst_device, const void* src_device, std::size_t bytes,
                    StreamHandle stream) {
  if (bytes == 0 || dst_device == src_device) return;
  check_cuda(cudaMemcpyAsync(dst_device, src_device, bytes, cudaMemcpyDeviceToDevice, stream),
             "cudaMemcpyAsync D2D");
}

void memset_zero_async(void* dst_device, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  check_cuda(cudaMemsetAsync(dst_device, 0, bytes, stream), "cudaMemsetAsync");
}

void synchronize_stream(StreamHandle stream) {
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

// ---------------------------------------------------------------------------
// CudaGraphExecutor
// ---------------------------------------------------------------------------

struct CudaGraphExecutor::Impl {
  struct Entry {
    cudaGraphExec_t exec = nullptr;
    bool warmed_up = false;
    bool uncapturable = false;
  };

  bool warmup_before_capture = true;
  cudaStream_t stream = nullptr;
  std::unordered_map<std::uint64_t, Entry> entries;
  std::size_t captures = 0;
  std::size_t replays = 0;
  std::size_t fallbacks = 0;

  explicit Impl(bool warmup) : warmup_before_capture(warmup) {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) {
      check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "CudaGraphExecutor cudaStreamCreateWithFlags");
    } else {
      (void)cudaGetLastError();
    }
  }

  void clear_entries() noexcept {
    for (auto& kv : entries) {
      if (kv.second.exec != nullptr) {
        (void)cudaGraphExecDestroy(kv.second.exec);
        kv.second.exec = nullptr;
      }
    }
    entries.clear();
  }

  ~Impl() {
    clear_entries();
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }
};

CudaGraphExecutor::CudaGraphExecutor(bool warmup_before_capture)
    : impl_(std::make_unique<Impl>(warmup_before_capture)) {}

CudaGraphExecutor::~CudaGraphExecutor() = default;
CudaGraphExecutor::CudaGraphExecutor(CudaGraphExecutor&&) noexcept = default;
CudaGraphExecutor& CudaGraphExecutor::operator=(CudaGraphExecutor&&) noexcept = default;

bool CudaGraphExecutor::capture(std::uint64_t graph_key,
                                const std::function<void(StreamHandle)>& record_fn) {
  if (impl_->stream == nullptr) {
    return false;
  }
  auto& entry = impl_->entries[graph_key];
  if (entry.exec != nullptr) {
    (void)cudaGraphExecDestroy(entry.exec);
    entry.exec = nullptr;
  }

  check_cuda(cudaStreamSynchronize(impl_->stream), "CudaGraphExecutor pre-capture sync");

  cudaError_t status = cudaStreamBeginCapture(impl_->stream, cudaStreamCaptureModeRelaxed);
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    entry.uncapturable = true;
    impl_->fallbacks += 1;
    return false;
  }

  bool fn_ok = true;
  try {
    record_fn(impl_->stream);
  } catch (...) {
    fn_ok = false;
  }

  cudaGraph_t graph = nullptr;
  cudaError_t end_status = cudaStreamEndCapture(impl_->stream, &graph);
  if (!fn_ok || end_status != cudaSuccess || graph == nullptr) {
    (void)cudaGetLastError();
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
    }
    entry.uncapturable = true;
    impl_->fallbacks += 1;
    return false;
  }

  cudaGraphExec_t exec = nullptr;
  cudaError_t inst_status = cudaGraphInstantiateWithFlags(&exec, graph, 0);
  (void)cudaGraphDestroy(graph);
  if (inst_status != cudaSuccess || exec == nullptr) {
    (void)cudaGetLastError();
    entry.uncapturable = true;
    impl_->fallbacks += 1;
    return false;
  }

  entry.exec = exec;
  entry.warmed_up = true;
  entry.uncapturable = false;
  impl_->captures += 1;
  return true;
}

void CudaGraphExecutor::launch(std::uint64_t graph_key) {
  auto it = impl_->entries.find(graph_key);
  if (it == impl_->entries.end() || it->second.exec == nullptr) {
    throw std::logic_error("CudaGraphExecutor::launch: graph_key has not been captured");
  }
  check_cuda(cudaGraphLaunch(it->second.exec, impl_->stream), "cudaGraphLaunch");
  impl_->replays += 1;
}

void CudaGraphExecutor::execute(std::uint64_t graph_key,
                                const std::function<void(StreamHandle)>& record_fn) {
  if (impl_->stream == nullptr) {
    record_fn(nullptr);
    return;
  }
  auto& entry = impl_->entries[graph_key];
  if (entry.exec != nullptr) {
    check_cuda(cudaGraphLaunch(entry.exec, impl_->stream), "cudaGraphLaunch");
    impl_->replays += 1;
    return;
  }
  if (entry.uncapturable) {
    record_fn(impl_->stream);
    impl_->fallbacks += 1;
    return;
  }
  if (impl_->warmup_before_capture && !entry.warmed_up) {
    record_fn(impl_->stream);
    entry.warmed_up = true;
    return;
  }
  if (capture(graph_key, record_fn)) {
    check_cuda(cudaGraphLaunch(entry.exec, impl_->stream), "cudaGraphLaunch");
    impl_->replays += 1;
  } else {
    record_fn(impl_->stream);
  }
}

bool CudaGraphExecutor::is_captured(std::uint64_t graph_key) const {
  auto it = impl_->entries.find(graph_key);
  return it != impl_->entries.end() && it->second.exec != nullptr;
}

void CudaGraphExecutor::invalidate(std::uint64_t graph_key) {
  auto it = impl_->entries.find(graph_key);
  if (it != impl_->entries.end()) {
    if (it->second.exec != nullptr) {
      (void)cudaGraphExecDestroy(it->second.exec);
    }
    impl_->entries.erase(it);
  }
}

void CudaGraphExecutor::clear() { impl_->clear_entries(); }

void CudaGraphExecutor::synchronize() {
  if (impl_->stream != nullptr) {
    check_cuda(cudaStreamSynchronize(impl_->stream), "CudaGraphExecutor::synchronize");
  }
}

StreamHandle CudaGraphExecutor::stream() const noexcept {
  return impl_ ? impl_->stream : nullptr;
}

std::size_t CudaGraphExecutor::capture_count() const noexcept { return impl_->captures; }
std::size_t CudaGraphExecutor::replay_count() const noexcept { return impl_->replays; }
std::size_t CudaGraphExecutor::fallback_count() const noexcept { return impl_->fallbacks; }

}  // namespace gpu4dft
