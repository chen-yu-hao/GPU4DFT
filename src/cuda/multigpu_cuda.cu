#include "gpu4dft/multigpu.hpp"

#include <cuda_runtime.h>

#if defined(GPU4DFT_HAVE_NCCL)
#include <nccl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu4dft::detail {

namespace {

inline void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

__global__ void set_int_kernel(int* ptr, int val) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *ptr = val;
  }
}

// Persistent thread-block dynamic work-stealing kernel: each block repeatedly claims
// the next task index via `claim_next_task(d_counter, 1)` until all `num_tasks` are processed.
__global__ void persistent_worksteal_kernel(int num_tasks, int* d_counter, const int* d_weights,
                                            double* d_out, int* d_block_counts) {
  __shared__ int s_task_id;
  __shared__ double s_partial[256];

  int claimed_by_block = 0;
  for (;;) {
    if (threadIdx.x == 0) {
      s_task_id = claim_next_task(d_counter, 1);
      if (s_task_id < num_tasks) {
        claimed_by_block += 1;
      }
    }
    __syncthreads();

    const int task = s_task_id;
    if (task >= num_tasks) break;

    const int iters = max(1, d_weights[task]);
    double acc = 0.0;
    for (int k = static_cast<int>(threadIdx.x); k < iters; k += static_cast<int>(blockDim.x)) {
      const double x = static_cast<double>(task + 1) * 1e-3 + static_cast<double>(k + 1) * 1e-5;
      acc += 1.0 / (1.0 + x * x);
    }
    s_partial[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      d_out[task] = s_partial[0];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && d_block_counts != nullptr) {
    d_block_counts[blockIdx.x] = claimed_by_block;
  }
}

__global__ void add_inplace_kernel(std::size_t count, const double* src, double* dst) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] += src[idx];
  }
}

}  // namespace

int cuda_visible_devices() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

void set_int_device(int* d_ptr, int value, StreamHandle stream) {
  set_int_kernel<<<1, 1, 0, stream>>>(d_ptr, value);
  check_cuda(cudaGetLastError(), "set_int_kernel");
}

std::vector<int> run_worksteal_cuda(int device_id, int* d_counter, int num_tasks,
                                    const int* h_weights, double* h_out, int num_blocks,
                                    StreamHandle stream) {
  if (num_tasks <= 0) return {};
  num_blocks = std::max(1, num_blocks);

  int prev_dev = 0;
  check_cuda(cudaGetDevice(&prev_dev), "cudaGetDevice");
  if (device_id >= 0 && device_id != prev_dev) {
    check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
  }

  auto& pool = default_device_pool(device_id >= 0 ? device_id : prev_dev);
  PooledBuffer<int> d_weights(pool, static_cast<std::size_t>(num_tasks), stream);
  PooledBuffer<double> d_out(pool, static_cast<std::size_t>(num_tasks), stream);
  PooledBuffer<int> d_block_counts(pool, static_cast<std::size_t>(num_blocks), stream);

  copy_h2d_async(d_weights.data(), h_weights, static_cast<std::size_t>(num_tasks) * sizeof(int),
                 stream);
  set_int_device(d_counter, 0, stream);

  persistent_worksteal_kernel<<<num_blocks, 256, 0, stream>>>(
      num_tasks, d_counter, d_weights.data(), d_out.data(), d_block_counts.data());
  check_cuda(cudaGetLastError(), "persistent_worksteal_kernel");

  std::vector<int> block_counts(static_cast<std::size_t>(num_blocks), 0);
  copy_d2h_async(h_out, d_out.data(), static_cast<std::size_t>(num_tasks) * sizeof(double), stream);
  copy_d2h_async(block_counts.data(), d_block_counts.data(),
                 static_cast<std::size_t>(num_blocks) * sizeof(int), stream);
  synchronize_stream(stream);

  if (device_id >= 0 && device_id != prev_dev) {
    (void)cudaSetDevice(prev_dev);
  }
  return block_counts;
}

void add_inplace_cuda(std::size_t count, const double* d_src, double* d_dst, StreamHandle stream) {
  if (count == 0) return;
  const int blocks = static_cast<int>((count + 255) / 256);
  add_inplace_kernel<<<blocks, 256, 0, stream>>>(count, d_src, d_dst);
  check_cuda(cudaGetLastError(), "add_inplace_kernel");
}

void allreduce_multi_gpu_cuda(const std::vector<int>& device_ids,
                              const std::vector<StreamHandle>& streams,
                              const std::vector<std::unique_ptr<DeviceMemoryPool>>& pools,
                              const std::vector<std::vector<bool>>& p2p_matrix,
                              std::span<double* const> per_gpu_d_bufs, std::size_t count) {
  const std::size_t ngpu = device_ids.size();
  if (ngpu <= 1 || count == 0) return;
  if (per_gpu_d_bufs.size() != ngpu) {
    throw std::invalid_argument(
        "MultiGpuManager::allreduce_sum: buffer span size must match num_devices()");
  }

#if defined(GPU4DFT_HAVE_NCCL)
  std::vector<ncclComm_t> comms(ngpu, nullptr);
  if (ncclCommInitAll(comms.data(), static_cast<int>(ngpu), device_ids.data()) == ncclSuccess) {
    ncclGroupStart();
    for (std::size_t r = 0; r < ngpu; ++r) {
      cudaSetDevice(device_ids[r]);
      ncclAllReduce(per_gpu_d_bufs[r], per_gpu_d_bufs[r], count, ncclDouble, ncclSum, comms[r],
                    streams[r]);
    }
    ncclGroupEnd();
    for (std::size_t r = 0; r < ngpu; ++r) {
      cudaSetDevice(device_ids[r]);
      cudaStreamSynchronize(streams[r]);
      ncclCommDestroy(comms[r]);
    }
    return;
  }
#endif

  int prev_dev = 0;
  check_cuda(cudaGetDevice(&prev_dev), "cudaGetDevice");

  // Synchronize peer streams before reading peer buffers
  for (std::size_t r = 0; r < ngpu; ++r) {
    check_cuda(cudaSetDevice(device_ids[r]), "cudaSetDevice");
    check_cuda(cudaStreamSynchronize(streams[r]), "cudaStreamSynchronize pre-allreduce");
  }

  // Reduce onto rank 0 via CUDA P2P (or host staging if P2P unavailable between PCIe roots)
  check_cuda(cudaSetDevice(device_ids[0]), "cudaSetDevice rank 0");
  PooledBuffer<double> d_tmp(*pools[0], count, streams[0]);
  std::vector<double> h_stage;

  for (std::size_t r = 1; r < ngpu; ++r) {
    if (p2p_matrix[0][r]) {
      check_cuda(cudaMemcpyPeerAsync(d_tmp.data(), device_ids[0], per_gpu_d_bufs[r], device_ids[r],
                                     count * sizeof(double), streams[0]),
                 "cudaMemcpyPeerAsync gather");
    } else {
      h_stage.resize(count);
      check_cuda(cudaSetDevice(device_ids[r]), "cudaSetDevice peer");
      check_cuda(cudaMemcpyAsync(h_stage.data(), per_gpu_d_bufs[r], count * sizeof(double),
                                 cudaMemcpyDeviceToHost, streams[r]),
                 "cudaMemcpyAsync D2H stage");
      check_cuda(cudaStreamSynchronize(streams[r]), "cudaStreamSynchronize stage");
      check_cuda(cudaSetDevice(device_ids[0]), "cudaSetDevice rank 0");
      check_cuda(cudaMemcpyAsync(d_tmp.data(), h_stage.data(), count * sizeof(double),
                                 cudaMemcpyHostToDevice, streams[0]),
                 "cudaMemcpyAsync H2D stage");
    }
    add_inplace_cuda(count, d_tmp.data(), per_gpu_d_bufs[0], streams[0]);
  }
  check_cuda(cudaStreamSynchronize(streams[0]), "cudaStreamSynchronize rank 0 sum");

  // Broadcast summed buffer from rank 0 back to ranks 1..G-1
  for (std::size_t r = 1; r < ngpu; ++r) {
    if (p2p_matrix[r][0]) {
      check_cuda(cudaMemcpyPeerAsync(per_gpu_d_bufs[r], device_ids[r], per_gpu_d_bufs[0],
                                     device_ids[0], count * sizeof(double), streams[r]),
                 "cudaMemcpyPeerAsync broadcast");
    } else {
      if (h_stage.size() != count) {
        h_stage.resize(count);
      }
      check_cuda(cudaSetDevice(device_ids[0]), "cudaSetDevice rank 0");
      check_cuda(cudaMemcpy(h_stage.data(), per_gpu_d_bufs[0], count * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H bcast");
      check_cuda(cudaSetDevice(device_ids[r]), "cudaSetDevice peer");
      check_cuda(cudaMemcpyAsync(per_gpu_d_bufs[r], h_stage.data(), count * sizeof(double),
                                 cudaMemcpyHostToDevice, streams[r]),
                 "cudaMemcpyAsync H2D bcast");
    }
  }

  for (std::size_t r = 1; r < ngpu; ++r) {
    check_cuda(cudaSetDevice(device_ids[r]), "cudaSetDevice sync");
    check_cuda(cudaStreamSynchronize(streams[r]), "cudaStreamSynchronize post-allreduce");
  }
  (void)cudaSetDevice(prev_dev);
}

}  // namespace gpu4dft::detail
