#include "gpu4dft/device_pool.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace gpu4dft::detail {

namespace {

inline void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

// Bank-conflict-free tiled R = M - M^T (since S D F = (F D S)^T for symmetric F, D, S).
__global__ void antisymmetrize_square_kernel(int n, const double* M, double* R_out) {
  __shared__ double tile_xy[32][33];
  __shared__ double tile_yx[32][33];

  const int x = static_cast<int>(blockIdx.x) * 32 + static_cast<int>(threadIdx.x);
  const int y = static_cast<int>(blockIdx.y) * 32 + static_cast<int>(threadIdx.y);
  if (x < n && y < n) {
    tile_xy[threadIdx.y][threadIdx.x] = M[static_cast<std::size_t>(y) * n + x];
  }

  const int tx = static_cast<int>(blockIdx.y) * 32 + static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(blockIdx.x) * 32 + static_cast<int>(threadIdx.y);
  if (tx < n && ty < n) {
    tile_yx[threadIdx.y][threadIdx.x] = M[static_cast<std::size_t>(ty) * n + tx];
  }
  __syncthreads();

  if (x < n && y < n) {
    // M[y, x] - M[x, y]
    R_out[static_cast<std::size_t>(y) * n + x] =
        tile_xy[threadIdx.y][threadIdx.x] - tile_yx[threadIdx.x][threadIdx.y];
  }
}

// Computes row_out[j] = <err_ring[slot], err_ring[j]> for j in [0, num_vecs).
__global__ void diis_dot_row_kernel(std::size_t err_len, int num_vecs, const double* err_ring,
                                    int slot, double* row_out) {
  const int j = static_cast<int>(blockIdx.y);
  if (j >= num_vecs) return;

  __shared__ double s_sum[256];
  const double* e_slot = err_ring + static_cast<std::size_t>(slot) * err_len;
  const double* e_j = err_ring + static_cast<std::size_t>(j) * err_len;

  double acc = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < err_len; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    acc += e_slot[idx] * e_j[idx];
  }
  s_sum[threadIdx.x] = acc;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(&row_out[j], s_sum[0]);
  }
}

// Fixed-topology linear combination F_out[idx] = sum_{k=0}^{max_vec-1} coeffs[k] * F_ring[k, idx].
__global__ void diis_extrapolate_kernel(std::size_t fock_len, int max_vec, const double* F_ring,
                                        const double* coeffs, double* F_out) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= fock_len) return;
  double sum = 0.0;
  for (int k = 0; k < max_vec; ++k) {
    const double c = coeffs[k];
    if (c != 0.0) {
      sum += c * F_ring[static_cast<std::size_t>(k) * fock_len + idx];
    }
  }
  F_out[idx] = sum;
}

}  // namespace

void diis_antisymmetrize_cuda(int n, const double* d_M, double* d_R_out, StreamHandle stream) {
  if (n <= 0) return;
  const dim3 block(32, 32);
  const dim3 grid((n + 31) / 32, (n + 31) / 32);
  antisymmetrize_square_kernel<<<grid, block, 0, stream>>>(n, d_M, d_R_out);
  check_cuda(cudaGetLastError(), "antisymmetrize_square_kernel");
}

void diis_dot_row_cuda(std::size_t err_len, int num_vecs, const double* d_err_ring, int slot,
                       double* d_row_out, StreamHandle stream) {
  if (num_vecs <= 0) return;
  memset_zero_async(d_row_out, static_cast<std::size_t>(num_vecs) * sizeof(double), stream);
  if (err_len == 0) return;
  const int blocks_x = static_cast<int>(std::min<std::size_t>((err_len + 255) / 256, 256));
  const dim3 grid(blocks_x, num_vecs);
  diis_dot_row_kernel<<<grid, 256, 0, stream>>>(err_len, num_vecs, d_err_ring, slot, d_row_out);
  check_cuda(cudaGetLastError(), "diis_dot_row_kernel");
}

void diis_extrapolate_cuda(std::size_t fock_len, int max_vec, const double* d_F_ring,
                           const double* d_coeffs, double* d_F_out, StreamHandle stream) {
  if (fock_len == 0) return;
  const int blocks = static_cast<int>((fock_len + 255) / 256);
  diis_extrapolate_kernel<<<blocks, 256, 0, stream>>>(fock_len, max_vec, d_F_ring, d_coeffs,
                                                      d_F_out);
  check_cuda(cudaGetLastError(), "diis_extrapolate_kernel");
}

}  // namespace gpu4dft::detail
