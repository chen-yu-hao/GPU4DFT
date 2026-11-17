#include "gpu4dft/pw.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpu4dft/generated/xc_kernels.hpp"

namespace gpu4dft::pw::detail {

static_assert(sizeof(cuDoubleComplex) == sizeof(Complex),
              "cuDoubleComplex and std::complex<double> must have identical layout");

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kFourPi = 4.0 * kPi;

inline void check_cuda(cudaError_t err, const char* ctx) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("PW CUDA error in ") + ctx + ": " +
                             cudaGetErrorString(err));
  }
}

inline void check_cublas(cublasStatus_t st, const char* ctx) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("PW cuBLAS error in ") + ctx + " (code " +
                             std::to_string(static_cast<int>(st)) + ")");
  }
}

inline void check_cusolver(cusolverStatus_t st, const char* ctx) {
  if (st != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("PW cuSOLVER error in ") + ctx + " (code " +
                             std::to_string(static_cast<int>(st)) + ")");
  }
}

inline void check_cufft(cufftResult st, const char* ctx) {
  if (st != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string("PW cuFFT error in ") + ctx + " (code " +
                             std::to_string(static_cast<int>(st)) + ")");
  }
}

// Thread-local best-fit GPU memory pool eliminating synchronous cudaMalloc/cudaFree
// device-wide stalls inside inner Davidson/LOBPCG and ACE loops.
struct GpuRawMemPool {
  std::vector<std::pair<std::size_t, void*>> free_blocks;

  ~GpuRawMemPool() {
    if (cudaFree(nullptr) == cudaErrorCudartUnloading) return;
    for (auto& blk : free_blocks) {
      if (blk.second != nullptr) cudaFree(blk.second);
    }
  }

  void* acquire(std::size_t bytes) {
    if (bytes == 0) return nullptr;
    const std::size_t aligned = ((bytes + 4095) / 4096) * 4096;
    std::size_t best_idx = free_blocks.size();
    std::size_t best_cap = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i < free_blocks.size(); ++i) {
      const std::size_t cap = free_blocks[i].first;
      if (cap >= aligned && cap <= aligned * 4 && cap < best_cap) {
        best_cap = cap;
        best_idx = i;
        if (cap == aligned) break;
      }
    }
    if (best_idx < free_blocks.size()) {
      void* ptr = free_blocks[best_idx].second;
      free_blocks[best_idx] = free_blocks.back();
      free_blocks.pop_back();
      return ptr;
    }
    void* ptr = nullptr;
    check_cuda(cudaMalloc(&ptr, aligned), "GpuRawMemPool::cudaMalloc");
    return ptr;
  }

  void release(void* ptr, std::size_t cap_bytes) noexcept {
    if (ptr == nullptr || cap_bytes == 0) return;
    const std::size_t aligned = ((cap_bytes + 4095) / 4096) * 4096;
    free_blocks.emplace_back(aligned, ptr);
  }

  static GpuRawMemPool& instance() {
    thread_local GpuRawMemPool pool;
    return pool;
  }
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t count) { resize(count); }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_), cap_(other.cap_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.cap_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      size_ = other.size_;
      cap_ = other.cap_;
      other.ptr_ = nullptr;
      other.size_ = 0;
      other.cap_ = 0;
    }
    return *this;
  }

  void resize(std::size_t count) {
    if (count > cap_) {
      reset();
      if (count > 0) {
        ptr_ = static_cast<T*>(GpuRawMemPool::instance().acquire(count * sizeof(T)));
        cap_ = count;
      }
    }
    size_ = count;
  }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      GpuRawMemPool::instance().release(ptr_, cap_ * sizeof(T));
      ptr_ = nullptr;
      size_ = 0;
      cap_ = 0;
    }
  }

  void zero() {
    if (size_ > 0) {
      check_cuda(cudaMemset(ptr_, 0, size_ * sizeof(T)), "DeviceBuffer::zero");
    }
  }

  void upload(const T* host_src, std::size_t count) {
    resize(count);
    if (count > 0) {
      check_cuda(cudaMemcpy(ptr_, host_src, count * sizeof(T), cudaMemcpyHostToDevice),
                 "DeviceBuffer::upload");
    }
  }

  void download(T* host_dst, std::size_t count) const {
    if (count > 0) {
      check_cuda(cudaMemcpy(host_dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost),
                 "DeviceBuffer::download");
    }
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return size_; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
  std::size_t cap_ = 0;
};

// SIMT FP64 tall-skinny complex GEMM kernels (avoids ~0.5..2.2 ms emulated z884 TensorOp
// split-K overhead in cuBLAS on sm_120 workstation Blackwell GPUs for small band counts m,n <= 64).
__global__ void pw_tall_zgemm_cn_kernel(int m,
                                        int n,
                                        int k,
                                        cuDoubleComplex alpha,
                                        const cuDoubleComplex* __restrict__ A,
                                        int lda,
                                        const cuDoubleComplex* __restrict__ B,
                                        int ldb,
                                        cuDoubleComplex beta,
                                        cuDoubleComplex* __restrict__ C,
                                        int ldc) {
  __shared__ double sx[128];
  __shared__ double sy[128];
  const int i = blockIdx.x;
  const int j = blockIdx.y;
  if (i >= m || j >= n) return;
  const int tid = threadIdx.x;
  const cuDoubleComplex* col_a = A + static_cast<std::size_t>(i) * lda;
  const cuDoubleComplex* col_b = B + static_cast<std::size_t>(j) * ldb;

  double sum_x = 0.0;
  double sum_y = 0.0;
  for (int p = tid; p < k; p += 128) {
    const cuDoubleComplex a = col_a[p];
    const cuDoubleComplex b = col_b[p];
    sum_x += a.x * b.x + a.y * b.y;
    sum_y += a.x * b.y - a.y * b.x;
  }
  sx[tid] = sum_x;
  sy[tid] = sum_y;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sx[tid] += sx[tid + stride];
      sy[tid] += sy[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    double rx = alpha.x * sx[0] - alpha.y * sy[0];
    double ry = alpha.x * sy[0] + alpha.y * sx[0];
    const std::size_t c_idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * ldc;
    if (beta.x != 0.0 || beta.y != 0.0) {
      const cuDoubleComplex c0 = C[c_idx];
      rx += beta.x * c0.x - beta.y * c0.y;
      ry += beta.x * c0.y + beta.y * c0.x;
    }
    C[c_idx] = make_cuDoubleComplex(rx, ry);
  }
}

__global__ void pw_tall_zgemm_nn_kernel(int m,
                                        int n,
                                        int k,
                                        cuDoubleComplex alpha,
                                        const cuDoubleComplex* __restrict__ A,
                                        int lda,
                                        const cuDoubleComplex* __restrict__ B,
                                        int ldb,
                                        cuDoubleComplex beta,
                                        cuDoubleComplex* __restrict__ C,
                                        int ldc) {
  __shared__ cuDoubleComplex s_B[64];
  const int tid = threadIdx.x;
  const int col = blockIdx.y;
  if (col >= n) return;
  if (tid < k) {
    s_B[tid] = B[tid + static_cast<std::size_t>(col) * ldb];
  }
  __syncthreads();

  const int row = blockIdx.x * 128 + tid;
  if (row >= m) return;
  double sum_x = 0.0;
  double sum_y = 0.0;
  for (int p = 0; p < k; ++p) {
    const cuDoubleComplex a = A[row + static_cast<std::size_t>(p) * lda];
    const cuDoubleComplex b = s_B[p];
    sum_x += a.x * b.x - a.y * b.y;
    sum_y += a.x * b.y + a.y * b.x;
  }
  double rx = alpha.x * sum_x - alpha.y * sum_y;
  double ry = alpha.x * sum_y + alpha.y * sum_x;
  const std::size_t c_idx = static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * ldc;
  if (beta.x != 0.0 || beta.y != 0.0) {
    const cuDoubleComplex c0 = C[c_idx];
    rx += beta.x * c0.x - beta.y * c0.y;
    ry += beta.x * c0.y + beta.y * c0.x;
  }
  C[c_idx] = make_cuDoubleComplex(rx, ry);
}

// Persistent thread-local CUDA handle + `cufftPlanMany` pool (avoids ~300 ms
// cuBLAS/cuSOLVER/cuFFT JIT initialization overhead on repeated calls).
struct PwCudaHandlePool {
  cublasHandle_t cublas = nullptr;
  cusolverDnHandle_t cusolver = nullptr;
  std::unordered_map<std::uint64_t, cufftHandle> plans;
  DeviceBuffer<int> d_info;
  DeviceBuffer<cuDoubleComplex> d_solver_work;

  PwCudaHandlePool() {
    check_cublas(cublasCreate(&cublas), "cublasCreate");
    check_cublas(cublasSetMathMode(cublas, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
    check_cusolver(cusolverDnCreate(&cusolver), "cusolverDnCreate");
    d_info.resize(1);
  }

  ~PwCudaHandlePool() {
    if (cudaFree(nullptr) == cudaErrorCudartUnloading) {
      // Driver is already unloading at process exit; skip explicit handle destruction.
      return;
    }
    for (auto& kv : plans) {
      cufftDestroy(kv.second);
    }
    if (cusolver != nullptr) cusolverDnDestroy(cusolver);
    if (cublas != nullptr) cublasDestroy(cublas);
  }

  static PwCudaHandlePool& instance() {
    thread_local PwCudaHandlePool pool;
    return pool;
  }
};

class PwCudaContext {
 public:
  explicit PwCudaContext(std::array<int, 3> fft_grid)
      : grid_(fft_grid), ngrid_(fft_grid[0] * fft_grid[1] * fft_grid[2]), pool_(PwCudaHandlePool::instance()) {}

  cublasHandle_t cublas() const noexcept { return pool_.cublas; }
  cusolverDnHandle_t cusolver() const noexcept { return pool_.cusolver; }
  int ngrid() const noexcept { return ngrid_; }

  cufftHandle get_plan(int nbatch) {
    const std::uint64_t key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(grid_[0])) << 48) |
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(grid_[1])) << 32) |
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(grid_[2])) << 16) |
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(nbatch));
    auto it = pool_.plans.find(key);
    if (it != pool_.plans.end()) return it->second;
    int dims[3] = {grid_[0], grid_[1], grid_[2]};
    cufftHandle plan{};
    check_cufft(cufftPlanMany(&plan, 3, dims, nullptr, 1, ngrid_, nullptr, 1, ngrid_,
                              CUFFT_Z2Z, nbatch),
                "cufftPlanMany");
    pool_.plans.emplace(key, plan);
    return plan;
  }

  void exec_fft3d(int nbatch, cuDoubleComplex* d_in, cuDoubleComplex* d_out, int direction) {
    if (nbatch <= 0) return;
    const int dir_flag = (direction < 0) ? CUFFT_FORWARD : CUFFT_INVERSE;
    int offset = 0;
    int rem = nbatch;
    while (rem > 0) {
      int step = 1;
      if (rem >= 32) {
        step = 32;
      } else if (rem >= 16) {
        step = 16;
      } else if (rem >= 8) {
        step = 8;
      } else if (rem >= 4) {
        step = 4;
      } else if (rem >= 2) {
        step = 2;
      }
      cufftHandle plan = get_plan(step);
      check_cufft(cufftExecZ2Z(plan,
                               d_in + static_cast<std::size_t>(offset) * ngrid_,
                               d_out + static_cast<std::size_t>(offset) * ngrid_,
                               dir_flag),
                  "cufftExecZ2Z");
      offset += step;
      rem -= step;
    }
  }

  void zgemm(cublasOperation_t opA,
             cublasOperation_t opB,
             int m,
             int n,
             int k,
             cuDoubleComplex alpha,
             const cuDoubleComplex* d_A,
             int lda,
             const cuDoubleComplex* d_B,
             int ldb,
             cuDoubleComplex beta,
             cuDoubleComplex* d_C,
             int ldc) {
    if (m <= 0 || n <= 0) return;
    if (opA == CUBLAS_OP_C && opB == CUBLAS_OP_N && m <= 64 && n <= 64 && k >= 0) {
      pw_tall_zgemm_cn_kernel<<<dim3(static_cast<unsigned int>(m), static_cast<unsigned int>(n)), 128>>>(
          m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
      return;
    }
    if (opA == CUBLAS_OP_N && opB == CUBLAS_OP_N && n <= 64 && k <= 64 && k >= 0) {
      pw_tall_zgemm_nn_kernel<<<dim3(static_cast<unsigned int>((m + 127) / 128), static_cast<unsigned int>(n)), 128>>>(
          m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
      return;
    }
    check_cublas(cublasZgemm(pool_.cublas, opA, opB, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc),
                 "cublasZgemm");
  }

  int* d_info() noexcept { return pool_.d_info.data(); }
  DeviceBuffer<cuDoubleComplex>& solver_work() noexcept { return pool_.d_solver_work; }

 private:
  std::array<int, 3> grid_{0, 0, 0};
  int ngrid_ = 0;
  PwCudaHandlePool& pool_;
};

// ============================================================================
// GPU Kernels for Plane-Wave DFT, Batched FFT Scatter/Gather, XC & ACE
// ============================================================================

__global__ void pw_hermitianize_kernel(int n, cuDoubleComplex* mat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * n) return;
  const int i = idx % n;
  const int j = idx / n;
  if (i > j) {
    const cuDoubleComplex aij = mat[i + j * n];
    const cuDoubleComplex aji = mat[j + i * n];
    const cuDoubleComplex avg = make_cuDoubleComplex(0.5 * (aij.x + aji.x), 0.5 * (aij.y - aji.y));
    mat[i + j * n] = avg;
    mat[j + i * n] = make_cuDoubleComplex(avg.x, -avg.y);
  } else if (i == j) {
    mat[i + i * n].y = 0.0;
  }
}

__global__ void pw_kinetic_kernel(int npw,
                                  int nband,
                                  const double* __restrict__ kin,
                                  const cuDoubleComplex* __restrict__ psi,
                                  cuDoubleComplex* __restrict__ hpsi) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npw * nband) return;
  const int ig = idx % npw;
  const double t = kin[ig];
  const cuDoubleComplex c = psi[idx];
  hpsi[idx] = make_cuDoubleComplex(t * c.x, t * c.y);
}

__global__ void pw_scatter_sphere_to_box_kernel(int npw,
                                                int nband,
                                                int ngrid,
                                                const int* __restrict__ fft_idx,
                                                const cuDoubleComplex* __restrict__ psi,
                                                cuDoubleComplex* __restrict__ box) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npw * nband) return;
  const int ig = idx % npw;
  const int b = idx / npw;
  box[static_cast<std::size_t>(b) * ngrid + fft_idx[ig]] = psi[idx];
}

__global__ void pw_pointwise_veff_kernel(int ngrid,
                                         int nband,
                                         double inv_ngrid,
                                         const double* __restrict__ veff_r,
                                         cuDoubleComplex* __restrict__ box) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(ngrid) * nband;
  if (idx >= total) return;
  const int ir = static_cast<int>(idx % static_cast<std::size_t>(ngrid));
  const double s = veff_r[ir] * inv_ngrid;
  cuDoubleComplex v = box[idx];
  box[idx] = make_cuDoubleComplex(v.x * s, v.y * s);
}

__global__ void pw_gather_add_box_to_sphere_kernel(int npw,
                                                   int nband,
                                                   int ngrid,
                                                   const int* __restrict__ fft_idx,
                                                   const cuDoubleComplex* __restrict__ box,
                                                   cuDoubleComplex* __restrict__ hpsi) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npw * nband) return;
  const int ig = idx % npw;
  const int b = idx / npw;
  const cuDoubleComplex g = box[static_cast<std::size_t>(b) * ngrid + fft_idx[ig]];
  cuDoubleComplex cur = hpsi[idx];
  hpsi[idx] = make_cuDoubleComplex(cur.x + g.x, cur.y + g.y);
}

__global__ void pw_gather_scaled_box_to_sphere_kernel(int npw,
                                                      int nband,
                                                      int ngrid,
                                                      double scale,
                                                      const int* __restrict__ fft_idx,
                                                      const cuDoubleComplex* __restrict__ box,
                                                      cuDoubleComplex* __restrict__ dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npw * nband) return;
  const int ig = idx % npw;
  const int b = idx / npw;
  const cuDoubleComplex g = box[static_cast<std::size_t>(b) * ngrid + fft_idx[ig]];
  dst[idx] = make_cuDoubleComplex(scale * g.x, scale * g.y);
}

__global__ void pw_accum_density_kernel(int ngrid,
                                        int n_occ,
                                        double wk_occ,
                                        const cuDoubleComplex* __restrict__ box,
                                        double* __restrict__ rho_r) {
  const int ir = blockIdx.x * blockDim.x + threadIdx.x;
  if (ir >= ngrid) return;
  double sum = 0.0;
  for (int b = 0; b < n_occ; ++b) {
    const cuDoubleComplex u = box[static_cast<std::size_t>(b) * ngrid + ir];
    sum += u.x * u.x + u.y * u.y;
  }
  rho_r[ir] += wk_occ * sum;
}

__global__ void pw_real_to_complex_kernel(int n, const double* __restrict__ r, cuDoubleComplex* __restrict__ c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = make_cuDoubleComplex(r[i], 0.0);
}

__global__ void pw_complex_to_real_kernel(int n, const cuDoubleComplex* __restrict__ c, double* __restrict__ r) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) r[i] = c[i].x;
}

__global__ void pw_scale_complex_kernel(std::size_t n, double s, cuDoubleComplex* __restrict__ c) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i].x *= s;
    c[i].y *= s;
  }
}

__global__ void pw_hartree_g_kernel(int ngrid,
                                    double vol,
                                    const double* __restrict__ g2,
                                    const cuDoubleComplex* __restrict__ rho_g,
                                    cuDoubleComplex* __restrict__ vh_g,
                                    double* __restrict__ d_eh_accum) {
  __shared__ double s_sum[256];
  const int tid = threadIdx.x;
  const int ig = blockIdx.x * blockDim.x + tid;

  double loc_eh = 0.0;
  if (ig < ngrid) {
    const double gg = g2[ig];
    if (gg > 1e-18) {
      const double fac = kFourPi / gg;
      const cuDoubleComplex rg = rho_g[ig];
      vh_g[ig] = make_cuDoubleComplex(fac * rg.x, fac * rg.y);
      loc_eh = 0.5 * vol * fac * (rg.x * rg.x + rg.y * rg.y);
    } else {
      vh_g[ig] = make_cuDoubleComplex(0.0, 0.0);
    }
  }
  s_sum[tid] = loc_eh;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s_sum[tid] += s_sum[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(d_eh_accum, s_sum[0]);
}

__global__ void pw_gga_grad_prep_kernel(int ngrid,
                                        const double* __restrict__ g_xyz,
                                        const cuDoubleComplex* __restrict__ rho_g,
                                        cuDoubleComplex* __restrict__ grad_buf) {
  const int ig = blockIdx.x * blockDim.x + threadIdx.x;
  if (ig >= ngrid) return;
  const cuDoubleComplex rg = rho_g[ig];
  for (int d = 0; d < 3; ++d) {
    const double gd = g_xyz[3 * ig + d];
    // i * gd * (rg.x + i * rg.y) = (-gd * rg.y) + i * (gd * rg.x)
    grad_buf[static_cast<std::size_t>(d) * ngrid + ig] =
        make_cuDoubleComplex(-gd * rg.y, gd * rg.x);
  }
}

__global__ void pw_eval_xc_gpu_kernel(int ngrid,
                                      int xc_func_id,
                                      int is_gga,
                                      double rho_cutoff,
                                      double dvol,
                                      const double* __restrict__ rho_r,
                                      cuDoubleComplex* __restrict__ grad_buf,
                                      double* __restrict__ vxc_r,
                                      double* __restrict__ d_exc_accum) {
  __shared__ double s_exc[256];
  const int tid = threadIdx.x;
  const int ir = blockIdx.x * blockDim.x + tid;

  double loc_exc = 0.0;
  if (ir < ngrid) {
    const double r = rho_r[ir];
    double rx = 0.0, ry = 0.0, rz = 0.0, sig = 0.0;
    if (is_gga) {
      rx = grad_buf[ir].x;
      ry = grad_buf[ngrid + ir].x;
      rz = grad_buf[2 * ngrid + ir].x;
      sig = rx * rx + ry * ry + rz * rz;
    }
    xc_gen::XcPointOut pt{};
    xc_gen::eval_xc_point(static_cast<xc_gen::XcFunctionalId>(xc_func_id),
                          r, sig, 0.0, 1, pt, rho_cutoff);
    if (r >= rho_cutoff) loc_exc = r * pt.zk * dvol;
    vxc_r[ir] = pt.vrho;
    if (is_gga) {
      const double two_vs = 2.0 * pt.vsigma;
      grad_buf[ir] = make_cuDoubleComplex(two_vs * rx, 0.0);
      grad_buf[ngrid + ir] = make_cuDoubleComplex(two_vs * ry, 0.0);
      grad_buf[2 * ngrid + ir] = make_cuDoubleComplex(two_vs * rz, 0.0);
    }
  }
  s_exc[tid] = loc_exc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s_exc[tid] += s_exc[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(d_exc_accum, s_exc[0]);
}

__global__ void pw_gga_div_prep_kernel(int ngrid,
                                       double inv_ngrid,
                                       const double* __restrict__ g_xyz,
                                       const cuDoubleComplex* __restrict__ grad_buf,
                                       cuDoubleComplex* __restrict__ div_g) {
  const int ig = blockIdx.x * blockDim.x + threadIdx.x;
  if (ig >= ngrid) return;
  const double gx = g_xyz[3 * ig + 0];
  const double gy = g_xyz[3 * ig + 1];
  const double gz = g_xyz[3 * ig + 2];
  const cuDoubleComplex hx = grad_buf[ig];
  const cuDoubleComplex hy = grad_buf[ngrid + ig];
  const cuDoubleComplex hz = grad_buf[2 * ngrid + ig];
  const double dot_re = (gx * hx.x + gy * hy.x + gz * hz.x) * inv_ngrid;
  const double dot_im = (gx * hx.y + gy * hy.y + gz * hz.y) * inv_ngrid;
  // -i * (dot_re + i * dot_im) = dot_im - i * dot_re
  div_g[ig] = make_cuDoubleComplex(dot_im, -dot_re);
}

__global__ void pw_assemble_veff_kernel(int ngrid,
                                        int add_gga_div,
                                        const double* __restrict__ vloc_ion_r,
                                        const double* __restrict__ vh_r,
                                        const cuDoubleComplex* __restrict__ gga_div_r,
                                        double* __restrict__ vxc_r,
                                        double* __restrict__ veff_r) {
  const int ir = blockIdx.x * blockDim.x + threadIdx.x;
  if (ir >= ngrid) return;
  double vxc = vxc_r[ir];
  if (add_gga_div) {
    vxc += gga_div_r[ir].x;
    vxc_r[ir] = vxc;
  }
  veff_r[ir] = vloc_ion_r[ir] + vh_r[ir] + vxc;
}

// 1 block per band: computes residual norm, band kinetic energy, and Teter-preconditioned residual!
__global__ void pw_band_residual_precond_kernel(int npw,
                                                const double* __restrict__ kin,
                                                const double* __restrict__ theta,
                                                const cuDoubleComplex* __restrict__ psi,
                                                const cuDoubleComplex* __restrict__ hpsi,
                                                cuDoubleComplex* __restrict__ r_prec_out,
                                                double* __restrict__ res_norms_out) {
  __shared__ double s_res2[256];
  __shared__ double s_ekin[256];
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const cuDoubleComplex* p_col = psi + static_cast<std::size_t>(b) * npw;
  const cuDoubleComplex* hp_col = hpsi + static_cast<std::size_t>(b) * npw;
  cuDoubleComplex* w_col = r_prec_out + static_cast<std::size_t>(b) * npw;
  const double eig = theta[b];

  double loc_r2 = 0.0;
  double loc_ek = 0.0;
  for (int ig = tid; ig < npw; ig += blockDim.x) {
    const cuDoubleComplex p = p_col[ig];
    const cuDoubleComplex hp = hp_col[ig];
    const double rx = hp.x - eig * p.x;
    const double ry = hp.y - eig * p.y;
    loc_r2 += rx * rx + ry * ry;
    loc_ek += kin[ig] * (p.x * p.x + p.y * p.y);
  }
  s_res2[tid] = loc_r2;
  s_ekin[tid] = loc_ek;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_res2[tid] += s_res2[tid + stride];
      s_ekin[tid] += s_ekin[tid + stride];
    }
    __syncthreads();
  }

  const double ekin_band = s_ekin[0];
  if (tid == 0) res_norms_out[b] = ::sqrt(s_res2[0]);

  for (int ig = tid; ig < npw; ig += blockDim.x) {
    const cuDoubleComplex p = p_col[ig];
    const cuDoubleComplex hp = hp_col[ig];
    const double k_tpa = teter_preconditioner(kin[ig], ekin_band);
    w_col[ig] = make_cuDoubleComplex(k_tpa * (hp.x - eig * p.x), k_tpa * (hp.y - eig * p.y));
  }
}

__global__ void pw_col_norms_kernel(int npw,
                                    const cuDoubleComplex* __restrict__ V,
                                    double* __restrict__ norms_out) {
  __shared__ double s_n2[256];
  const int j = blockIdx.x;
  const int tid = threadIdx.x;
  const cuDoubleComplex* col = V + static_cast<std::size_t>(j) * npw;
  double loc = 0.0;
  for (int i = tid; i < npw; i += blockDim.x) {
    const cuDoubleComplex v = col[i];
    loc += v.x * v.x + v.y * v.y;
  }
  s_n2[tid] = loc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s_n2[tid] += s_n2[tid + stride];
    __syncthreads();
  }
  if (tid == 0) norms_out[j] = ::sqrt(s_n2[0]);
}

__global__ void pw_scale_selected_cols_kernel(int npw,
                                              int n_keep,
                                              const int* __restrict__ src_cols,
                                              const double* __restrict__ inv_norms,
                                              const cuDoubleComplex* __restrict__ V_src,
                                              cuDoubleComplex* __restrict__ V_dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npw * n_keep) return;
  const int ig = idx % npw;
  const int k = idx / npw;
  const int src_j = src_cols[k];
  const double s = inv_norms[k];
  const cuDoubleComplex v = V_src[static_cast<std::size_t>(src_j) * npw + ig];
  V_dst[idx] = make_cuDoubleComplex(s * v.x, s * v.y);
}

// Exact Exchange pair formation & accumulation GPU kernels
__global__ void pw_form_rect_pairs_kernel(int ngrid,
                                          int i_start,
                                          int n_i,
                                          int n_b,
                                          const cuDoubleComplex* __restrict__ u_occ,
                                          const cuDoubleComplex* __restrict__ u_trial,
                                          cuDoubleComplex* __restrict__ pair_buf) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t npairs = static_cast<std::size_t>(n_i) * n_b;
  if (idx >= npairs * ngrid) return;
  const int ir = static_cast<int>(idx % ngrid);
  const int p = static_cast<int>(idx / ngrid);
  const int i_loc = p / n_b;
  const int b = p % n_b;
  const cuDoubleComplex ui = u_occ[static_cast<std::size_t>(i_start + i_loc) * ngrid + ir];
  const cuDoubleComplex ub = u_trial[static_cast<std::size_t>(b) * ngrid + ir];
  // conj(ui) * ub
  pair_buf[idx] = make_cuDoubleComplex(ui.x * ub.x + ui.y * ub.y, ui.x * ub.y - ui.y * ub.x);
}

__global__ void pw_mul_coulomb_kernel(int ngrid,
                                      int npairs,
                                      double inv_ngrid,
                                      const double* __restrict__ coul_g,
                                      cuDoubleComplex* __restrict__ pair_buf) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::size_t>(npairs) * ngrid) return;
  const int ig = static_cast<int>(idx % ngrid);
  const double s = coul_g[ig] * inv_ngrid;
  pair_buf[idx].x *= s;
  pair_buf[idx].y *= s;
}

__global__ void pw_accum_rect_exx_kernel(int ngrid,
                                         int i_start,
                                         int n_i,
                                         int n_b,
                                         double wq,
                                         const cuDoubleComplex* __restrict__ u_occ,
                                         const cuDoubleComplex* __restrict__ pair_buf,
                                         cuDoubleComplex* __restrict__ w_r) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_b) * ngrid) return;
  const int ir = static_cast<int>(idx % ngrid);
  const int b = static_cast<int>(idx / ngrid);
  double acc_x = 0.0;
  double acc_y = 0.0;
  for (int i_loc = 0; i_loc < n_i; ++i_loc) {
    const cuDoubleComplex ui = u_occ[static_cast<std::size_t>(i_start + i_loc) * ngrid + ir];
    const cuDoubleComplex phi =
        pair_buf[static_cast<std::size_t>(i_loc * n_b + b) * ngrid + ir];
    acc_x += ui.x * phi.x - ui.y * phi.y;
    acc_y += ui.x * phi.y + ui.y * phi.x;
  }
  w_r[idx].x -= wq * acc_x;
  w_r[idx].y -= wq * acc_y;
}

__global__ void pw_form_symm_pairs_kernel(int ngrid,
                                          int n_occ,
                                          const cuDoubleComplex* __restrict__ u_occ,
                                          cuDoubleComplex* __restrict__ pair_buf) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int npairs = n_occ * (n_occ + 1) / 2;
  if (idx >= static_cast<std::size_t>(npairs) * ngrid) return;
  const int ir = static_cast<int>(idx % ngrid);
  const int p = static_cast<int>(idx / ngrid);
  int i = 0;
  int rem = p;
  for (int row = 0; row < n_occ; ++row) {
    const int row_len = n_occ - row;
    if (rem < row_len) {
      i = row;
      break;
    }
    rem -= row_len;
  }
  const int j = i + rem;
  const cuDoubleComplex ui = u_occ[static_cast<std::size_t>(i) * ngrid + ir];
  const cuDoubleComplex uj = u_occ[static_cast<std::size_t>(j) * ngrid + ir];
  pair_buf[idx] = make_cuDoubleComplex(ui.x * uj.x + ui.y * uj.y, ui.x * uj.y - ui.y * uj.x);
}

__global__ void pw_accum_symm_exx_kernel(int ngrid,
                                         int n_occ,
                                         double wq,
                                         const cuDoubleComplex* __restrict__ u_occ,
                                         const cuDoubleComplex* __restrict__ pair_buf,
                                         cuDoubleComplex* __restrict__ w_r) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::size_t>(n_occ) * ngrid) return;
  const int ir = static_cast<int>(idx % ngrid);
  const int b = static_cast<int>(idx / ngrid);
  double acc_x = 0.0;
  double acc_y = 0.0;
  for (int i = 0; i <= b; ++i) {
    const int p = i * n_occ - i * (i - 1) / 2 + (b - i);
    const cuDoubleComplex ui = u_occ[static_cast<std::size_t>(i) * ngrid + ir];
    const cuDoubleComplex phi = pair_buf[static_cast<std::size_t>(p) * ngrid + ir];
    acc_x += ui.x * phi.x - ui.y * phi.y;
    acc_y += ui.x * phi.y + ui.y * phi.x;
  }
  const int row_base = b * n_occ - b * (b - 1) / 2;
  for (int j = b + 1; j < n_occ; ++j) {
    const int p = row_base + (j - b);
    const cuDoubleComplex uj = u_occ[static_cast<std::size_t>(j) * ngrid + ir];
    const cuDoubleComplex phi = pair_buf[static_cast<std::size_t>(p) * ngrid + ir];
    acc_x += uj.x * phi.x + uj.y * phi.y;
    acc_y += -uj.x * phi.y + uj.y * phi.x;
  }
  w_r[idx].x -= wq * acc_x;
  w_r[idx].y -= wq * acc_y;
}

__global__ void pw_gather_cols_kernel(int npw,
                                      int n_sel,
                                      const int* __restrict__ sel_cols,
                                      const cuDoubleComplex* __restrict__ src,
                                      cuDoubleComplex* __restrict__ dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= npw * n_sel) return;
  const int ig = idx % npw;
  const int k = idx / npw;
  dst[idx] = src[static_cast<std::size_t>(sel_cols[k]) * npw + ig];
}

__global__ void pw_ekin_bands_kernel(int npw,
                                     int n_occ,
                                     double wk_occ,
                                     const double* __restrict__ kin,
                                     const cuDoubleComplex* __restrict__ psi,
                                     double* __restrict__ ekin_accum) {
  __shared__ double s_sum[256];
  const int tid = threadIdx.x;
  const int b = blockIdx.x;
  const cuDoubleComplex* col = psi + static_cast<std::size_t>(b) * npw;
  double loc = 0.0;
  if (b < n_occ) {
    for (int ig = tid; ig < npw; ig += blockDim.x) {
      const cuDoubleComplex c = col[ig];
      loc += kin[ig] * (c.x * c.x + c.y * c.y);
    }
  }
  s_sum[tid] = loc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s_sum[tid] += s_sum[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(ekin_accum, wk_occ * s_sum[0]);
}

// ============================================================================
// GPU Linear Algebra & Iterative Solvers (cuSOLVER Zheevd + Zpotrf + SVQB)
// ============================================================================

// Column-major Hermitian eigendecomposition d_A * U = U * diag(w) via cusolverDnZheevd.
// Overwrites `d_A` [n x n] with eigenvectors `U` and writes ascending eigenvalues into `d_w` [n].
void zheevd_gpu(PwCudaContext& ctx, int n, cuDoubleComplex* d_A, double* d_w) {
  if (n <= 0) return;
  if (n == 1) {
    Complex h_a{};
    check_cuda(cudaMemcpy(&h_a, d_A, sizeof(Complex), cudaMemcpyDeviceToHost), "zheevd_gpu n=1");
    const double w0 = h_a.real();
    const Complex one(1.0, 0.0);
    check_cuda(cudaMemcpy(d_w, &w0, sizeof(double), cudaMemcpyHostToDevice), "zheevd_gpu w0");
    check_cuda(cudaMemcpy(d_A, &one, sizeof(Complex), cudaMemcpyHostToDevice), "zheevd_gpu u0");
    return;
  }
  if (n <= 64) {
    // Subspace matrices n <= 64 (< 64 KiB): avoid ~1.6 ms cuSOLVER host-sync launch overhead
    // while keeping all [npw x n] projections on cuBLAS.
    std::vector<Complex> h_A(static_cast<std::size_t>(n) * n);
    std::vector<Complex> h_U(static_cast<std::size_t>(n) * n);
    std::vector<double> h_w(static_cast<std::size_t>(n));
    check_cuda(cudaMemcpy(h_A.data(), d_A, h_A.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
               "zheevd_gpu small download");
    pw_zheev_host(n, h_A.data(), h_w.data(), h_U.data());
    check_cuda(cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(double), cudaMemcpyHostToDevice),
               "zheevd_gpu small w upload");
    check_cuda(cudaMemcpy(d_A, h_U.data(), h_U.size() * sizeof(Complex), cudaMemcpyHostToDevice),
               "zheevd_gpu small U upload");
    return;
  }

  const int threads = 256;
  const int blocks = (n * n + threads - 1) / threads;
  pw_hermitianize_kernel<<<blocks, threads>>>(n, d_A);

  int lwork = 0;
  check_cusolver(cusolverDnZheevd_bufferSize(
                     ctx.cusolver(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                     n, d_A, n, d_w, &lwork),
                 "cusolverDnZheevd_bufferSize");
  ctx.solver_work().resize(static_cast<std::size_t>(std::max(1, lwork)));
  check_cusolver(cusolverDnZheevd(
                     ctx.cusolver(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                     n, d_A, n, d_w, ctx.solver_work().data(), lwork, ctx.d_info()),
                 "cusolverDnZheevd");
  int h_info = 0;
  check_cuda(cudaMemcpy(&h_info, ctx.d_info(), sizeof(int), cudaMemcpyDeviceToHost),
             "cusolverDnZheevd info");
  if (h_info != 0) {
    throw std::runtime_error("cusolverDnZheevd failed with info = " + std::to_string(h_info));
  }
}

// GPU Cholesky B = L * L^H (`cusolverDnZpotrf`) + right triangular solve Xi * L^H = W (`cublasZtrsm`)
void cholesky_and_trsm_ace_gpu(PwCudaContext& ctx,
                               int npw,
                               int n_occ,
                               cuDoubleComplex* d_B,
                               const cuDoubleComplex* d_W,
                               cuDoubleComplex* d_Xi) {
  if (n_occ <= 0) return;
  if (n_occ <= 64) {
    std::vector<Complex> h_B(static_cast<std::size_t>(n_occ) * n_occ);
    std::vector<Complex> h_I(static_cast<std::size_t>(n_occ) * n_occ, Complex(0.0, 0.0));
    std::vector<Complex> h_LinvH(static_cast<std::size_t>(n_occ) * n_occ);
    for (int i = 0; i < n_occ; ++i) h_I[i + static_cast<std::size_t>(i) * n_occ] = Complex(1.0, 0.0);
    check_cuda(cudaMemcpy(h_B.data(), d_B, h_B.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
               "cholesky small B download");
    pw_cholesky_trsm_ace_host(n_occ, n_occ, h_B.data(), h_I.data(), h_LinvH.data());
    DeviceBuffer<cuDoubleComplex> d_LinvH;
    d_LinvH.upload(reinterpret_cast<const cuDoubleComplex*>(h_LinvH.data()), h_LinvH.size());
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_occ, n_occ,
              make_cuDoubleComplex(1.0, 0.0), d_W, npw, d_LinvH.data(), n_occ,
              make_cuDoubleComplex(0.0, 0.0), d_Xi, npw);
    return;
  }
  const int threads = 256;
  const int blocks = (n_occ * n_occ + threads - 1) / threads;
  pw_hermitianize_kernel<<<blocks, threads>>>(n_occ, d_B);

  DeviceBuffer<cuDoubleComplex> d_B_copy(static_cast<std::size_t>(n_occ) * n_occ);
  check_cuda(cudaMemcpy(d_B_copy.data(), d_B,
                        static_cast<std::size_t>(n_occ) * n_occ * sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToDevice),
             "cholesky B backup");

  int lwork = 0;
  check_cusolver(cusolverDnZpotrf_bufferSize(
                     ctx.cusolver(), CUBLAS_FILL_MODE_LOWER, n_occ, d_B, n_occ, &lwork),
                 "cusolverDnZpotrf_bufferSize");
  ctx.solver_work().resize(static_cast<std::size_t>(std::max(1, lwork)));
  check_cusolver(cusolverDnZpotrf(ctx.cusolver(), CUBLAS_FILL_MODE_LOWER, n_occ, d_B, n_occ,
                                  ctx.solver_work().data(), lwork, ctx.d_info()),
                 "cusolverDnZpotrf");
  int h_info = 0;
  check_cuda(cudaMemcpy(&h_info, ctx.d_info(), sizeof(int), cudaMemcpyDeviceToHost),
             "cusolverDnZpotrf info");

  if (h_info == 0) {
    check_cuda(cudaMemcpy(d_Xi, d_W,
                          static_cast<std::size_t>(npw) * n_occ * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToDevice),
               "copy W -> Xi");
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    check_cublas(cublasZtrsm(ctx.cublas(), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT,
                             npw, n_occ, &one, d_B, n_occ, d_Xi, npw),
                 "cublasZtrsm ACE");
    return;
  }

  // Eigenvalue inverse-sqrt fallback if near-semidefinite
  DeviceBuffer<double> d_evals(static_cast<std::size_t>(n_occ));
  zheevd_gpu(ctx, n_occ, d_B_copy.data(), d_evals.data());
  std::vector<double> h_evals(static_cast<std::size_t>(n_occ));
  std::vector<Complex> h_evecs(static_cast<std::size_t>(n_occ) * n_occ);
  d_evals.download(h_evals.data(), static_cast<std::size_t>(n_occ));
  check_cuda(cudaMemcpy(h_evecs.data(), d_B_copy.data(),
                        h_evecs.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
             "download evecs fallback");
  std::vector<Complex> h_inv_s(static_cast<std::size_t>(n_occ) * n_occ, Complex(0.0, 0.0));
  for (int k = 0; k < n_occ; ++k) {
    if (h_evals[k] > 1e-13) {
      const double s = 1.0 / std::sqrt(h_evals[k]);
      for (int i = 0; i < n_occ; ++i) {
        h_inv_s[i + static_cast<std::size_t>(k) * n_occ] =
            h_evecs[i + static_cast<std::size_t>(k) * n_occ] * s;
      }
    }
  }
  DeviceBuffer<cuDoubleComplex> d_inv_s;
  d_inv_s.upload(reinterpret_cast<const cuDoubleComplex*>(h_inv_s.data()), h_inv_s.size());
  ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_occ, n_occ,
            make_cuDoubleComplex(1.0, 0.0), d_W, npw, d_inv_s.data(), n_occ,
            make_cuDoubleComplex(0.0, 0.0), d_Xi, npw);
}

// GPU Kahan two-pass Gram-Schmidt + SVQB orthonormalization
int svqb_orthonormalize_gpu(PwCudaContext& ctx,
                            int npw,
                            cuDoubleComplex* d_V,
                            int ncols,
                            const cuDoubleComplex* d_V_against,
                            int n_against,
                            cuDoubleComplex* d_HV_sync = nullptr,
                            const cuDoubleComplex* d_HV_against = nullptr) {
  if (ncols <= 0) return 0;
  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

  if (d_V_against != nullptr && n_against > 0) {
    DeviceBuffer<cuDoubleComplex> d_C(static_cast<std::size_t>(n_against) * ncols);
    for (int pass = 0; pass < 2; ++pass) {
      ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, n_against, ncols, npw, one,
                d_V_against, npw, d_V, npw, zero, d_C.data(), n_against);
      ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, ncols, n_against, neg_one,
                d_V_against, npw, d_C.data(), n_against, one, d_V, npw);
      if (d_HV_sync != nullptr && d_HV_against != nullptr) {
        ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, ncols, n_against, neg_one,
                  d_HV_against, npw, d_C.data(), n_against, one, d_HV_sync, npw);
      }
    }
  }

  if (ncols <= 64) {
    DeviceBuffer<cuDoubleComplex> d_G(static_cast<std::size_t>(ncols) * ncols);
    ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, ncols, ncols, npw, one,
              d_V, npw, d_V, npw, zero, d_G.data(), ncols);
    std::vector<Complex> h_G(static_cast<std::size_t>(ncols) * ncols);
    check_cuda(cudaMemcpy(h_G.data(), d_G.data(), h_G.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
               "svqb download G");

    std::vector<int> keep_cols;
    std::vector<double> inv_norms;
    for (int j = 0; j < ncols; ++j) {
      const double n2 = h_G[j + static_cast<std::size_t>(j) * ncols].real();
      if (n2 > 1e-26) {
        keep_cols.push_back(j);
        inv_norms.push_back(1.0 / std::sqrt(n2));
      }
    }
    const int n_keep = static_cast<int>(keep_cols.size());
    if (n_keep == 0) return 0;

    std::vector<Complex> h_S(static_cast<std::size_t>(n_keep) * n_keep);
    for (int b = 0; b < n_keep; ++b) {
      for (int a = 0; a < n_keep; ++a) {
        h_S[a + static_cast<std::size_t>(b) * n_keep] =
            h_G[keep_cols[a] + static_cast<std::size_t>(keep_cols[b]) * ncols] *
            (inv_norms[a] * inv_norms[b]);
      }
    }
    std::vector<double> h_svals(static_cast<std::size_t>(n_keep));
    std::vector<Complex> h_Z(static_cast<std::size_t>(n_keep) * n_keep);
    pw_zheev_host(n_keep, h_S.data(), h_svals.data(), h_Z.data());

    int n_out = 0;
    std::vector<Complex> h_T(static_cast<std::size_t>(ncols) * n_keep, Complex(0.0, 0.0));
    for (int j = 0; j < n_keep; ++j) {
      if (h_svals[j] > 1e-11) {
        const double inv_s = 1.0 / std::sqrt(h_svals[j]);
        for (int a = 0; a < n_keep; ++a) {
          h_T[keep_cols[a] + static_cast<std::size_t>(n_out) * ncols] =
              h_Z[a + static_cast<std::size_t>(j) * n_keep] * (inv_norms[a] * inv_s);
        }
        ++n_out;
      }
    }
    if (n_out == 0) return 0;

    DeviceBuffer<cuDoubleComplex> d_T;
    d_T.upload(reinterpret_cast<const cuDoubleComplex*>(h_T.data()),
               static_cast<std::size_t>(ncols) * n_out);
    DeviceBuffer<cuDoubleComplex> d_V_out(static_cast<std::size_t>(npw) * n_out);
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_out, ncols, one,
              d_V, npw, d_T.data(), ncols, zero, d_V_out.data(), npw);
    check_cuda(cudaMemcpy(d_V, d_V_out.data(),
                          static_cast<std::size_t>(npw) * n_out * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToDevice),
               "svqb copy V_out");
    if (d_HV_sync != nullptr) {
      ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_out, ncols, one,
                d_HV_sync, npw, d_T.data(), ncols, zero, d_V_out.data(), npw);
      check_cuda(cudaMemcpy(d_HV_sync, d_V_out.data(),
                            static_cast<std::size_t>(npw) * n_out * sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToDevice),
                 "svqb copy HV_out");
    }
    return n_out;
  }

  DeviceBuffer<double> d_norms(static_cast<std::size_t>(ncols));
  pw_col_norms_kernel<<<ncols, 256>>>(npw, d_V, d_norms.data());
  std::vector<double> h_norms(static_cast<std::size_t>(ncols));
  d_norms.download(h_norms.data(), static_cast<std::size_t>(ncols));

  std::vector<int> keep_cols;
  std::vector<double> inv_norms;
  for (int j = 0; j < ncols; ++j) {
    if (h_norms[j] > 1e-13) {
      keep_cols.push_back(j);
      inv_norms.push_back(1.0 / h_norms[j]);
    }
  }
  const int n_keep = static_cast<int>(keep_cols.size());
  if (n_keep == 0) return 0;

  DeviceBuffer<int> d_keep;
  DeviceBuffer<double> d_inv_norms;
  d_keep.upload(keep_cols.data(), keep_cols.size());
  d_inv_norms.upload(inv_norms.data(), inv_norms.size());

  DeviceBuffer<cuDoubleComplex> d_V_scaled(static_cast<std::size_t>(npw) * n_keep);
  const int threads = 256;
  const int blocks = (npw * n_keep + threads - 1) / threads;
  pw_scale_selected_cols_kernel<<<blocks, threads>>>(
      npw, n_keep, d_keep.data(), d_inv_norms.data(), d_V, d_V_scaled.data());

  DeviceBuffer<cuDoubleComplex> d_HV_scaled;
  if (d_HV_sync != nullptr) {
    d_HV_scaled.resize(static_cast<std::size_t>(npw) * n_keep);
    pw_scale_selected_cols_kernel<<<blocks, threads>>>(
        npw, n_keep, d_keep.data(), d_inv_norms.data(), d_HV_sync, d_HV_scaled.data());
  }

  DeviceBuffer<cuDoubleComplex> d_S(static_cast<std::size_t>(n_keep) * n_keep);
  ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, n_keep, n_keep, npw, one,
            d_V_scaled.data(), npw, d_V_scaled.data(), npw, zero, d_S.data(), n_keep);

  DeviceBuffer<double> d_svals(static_cast<std::size_t>(n_keep));
  zheevd_gpu(ctx, n_keep, d_S.data(), d_svals.data());

  std::vector<double> h_svals(static_cast<std::size_t>(n_keep));
  std::vector<Complex> h_Z(static_cast<std::size_t>(n_keep) * n_keep);
  d_svals.download(h_svals.data(), h_svals.size());
  check_cuda(cudaMemcpy(h_Z.data(), d_S.data(), h_Z.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
             "svqb download Z");

  int n_out = 0;
  std::vector<Complex> h_T(static_cast<std::size_t>(n_keep) * n_keep, Complex(0.0, 0.0));
  for (int j = 0; j < n_keep; ++j) {
    if (h_svals[j] > 1e-11) {
      const double inv_s = 1.0 / std::sqrt(h_svals[j]);
      for (int i = 0; i < n_keep; ++i) {
        h_T[i + static_cast<std::size_t>(n_out) * n_keep] =
            h_Z[i + static_cast<std::size_t>(j) * n_keep] * inv_s;
      }
      ++n_out;
    }
  }
  if (n_out == 0) return 0;

  DeviceBuffer<cuDoubleComplex> d_T;
  d_T.upload(reinterpret_cast<const cuDoubleComplex*>(h_T.data()), h_T.size());
  ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_out, n_keep, one,
            d_V_scaled.data(), npw, d_T.data(), n_keep, zero, d_V, npw);
  if (d_HV_sync != nullptr) {
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_out, n_keep, one,
              d_HV_scaled.data(), npw, d_T.data(), n_keep, zero, d_HV_sync, npw);
  }
  return n_out;
}

// Pre-uploaded GPU per-k-point Hamiltonian state
struct GpuKPointData {
  int npw = 0;
  int nproj = 0;
  double weight = 1.0;
  DeviceBuffer<int> d_fft_idx;
  DeviceBuffer<double> d_kin;
  DeviceBuffer<cuDoubleComplex> d_P;
  DeviceBuffer<cuDoubleComplex> d_H_proj;
  DeviceBuffer<cuDoubleComplex> d_ace_xi;
};

void upload_kpoint_data(const PwCell& cell, const KPointBasis& kp, bool with_nl, GpuKPointData& out) {
  out.npw = kp.npw;
  out.weight = kp.weight;
  out.d_fft_idx.upload(kp.fft_indices.data(), kp.fft_indices.size());
  out.d_kin.upload(kp.kin.data(), kp.kin.size());
  out.nproj = 0;
  if (with_nl) {
    NonLocalProjector nl(cell, kp);
    out.nproj = nl.nproj();
    if (out.nproj > 0) {
      out.d_P.upload(reinterpret_cast<const cuDoubleComplex*>(nl.projectors_col_major().data()),
                     nl.projectors_col_major().size());
      out.d_H_proj.upload(
          reinterpret_cast<const cuDoubleComplex*>(nl.coupling_matrix_col_major().data()),
          nl.coupling_matrix_col_major().size());
    }
  }
}

// GPU Hamiltonian Action: Kinetic + Batched 3D cuFFT Local Potential + Batched cuBLAS ZGEMM V_NL + Batched cuBLAS ZGEMM ACE!
void apply_pw_hamiltonian_gpu(PwCudaContext& ctx,
                              const GpuKPointData& kp_gpu,
                              const double* d_veff_r,
                              int ace_nocc,
                              double alpha_x,
                              const cuDoubleComplex* d_psi,
                              cuDoubleComplex* d_hpsi,
                              int nband,
                              DeviceBuffer<cuDoubleComplex>& d_fft_work,
                              DeviceBuffer<cuDoubleComplex>& d_proj_work) {
  if (nband <= 0) return;
  const int npw = kp_gpu.npw;
  const int ngrid = ctx.ngrid();
  const double inv_ngrid = 1.0 / static_cast<double>(ngrid);
  const int threads = 256;

  // 1. Diagonal kinetic operator
  pw_kinetic_kernel<<<(npw * nband + threads - 1) / threads, threads>>>(
      npw, nband, kp_gpu.d_kin.data(), d_psi, d_hpsi);

  // 2. Batched 3D cuFFT (`cufftPlanMany`) for V_eff(r) * psi(r)
  const int chunk = std::min(nband, 32);
  d_fft_work.resize(static_cast<std::size_t>(chunk) * ngrid);
  for (int b0 = 0; b0 < nband; b0 += chunk) {
    const int cur_b = std::min(chunk, nband - b0);
    check_cuda(cudaMemset(d_fft_work.data(), 0,
                          static_cast<std::size_t>(cur_b) * ngrid * sizeof(cuDoubleComplex)),
               "memset fft_work");
    pw_scatter_sphere_to_box_kernel<<<(npw * cur_b + threads - 1) / threads, threads>>>(
        npw, cur_b, ngrid, kp_gpu.d_fft_idx.data(),
        d_psi + static_cast<std::size_t>(b0) * npw, d_fft_work.data());

    ctx.exec_fft3d(cur_b, d_fft_work.data(), d_fft_work.data(), +1);

    const std::size_t tot_pts = static_cast<std::size_t>(cur_b) * ngrid;
    pw_pointwise_veff_kernel<<<static_cast<unsigned int>((tot_pts + threads - 1) / threads), threads>>>(
        ngrid, cur_b, inv_ngrid, d_veff_r, d_fft_work.data());

    ctx.exec_fft3d(cur_b, d_fft_work.data(), d_fft_work.data(), -1);

    pw_gather_add_box_to_sphere_kernel<<<(npw * cur_b + threads - 1) / threads, threads>>>(
        npw, cur_b, ngrid, kp_gpu.d_fft_idx.data(), d_fft_work.data(),
        d_hpsi + static_cast<std::size_t>(b0) * npw);
  }

  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

  // 3. Non-local GTH projectors via two cublasZgemm calls (+ H_proj coupling)
  if (kp_gpu.nproj > 0) {
    const int nproj = kp_gpu.nproj;
    d_proj_work.resize(static_cast<std::size_t>(2 * nproj) * nband);
    cuDoubleComplex* d_C = d_proj_work.data();
    cuDoubleComplex* d_D = d_proj_work.data() + static_cast<std::size_t>(nproj) * nband;
    ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, nproj, nband, npw, one,
              kp_gpu.d_P.data(), npw, d_psi, npw, zero, d_C, nproj);
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, nproj, nband, nproj, one,
              kp_gpu.d_H_proj.data(), nproj, d_C, nproj, zero, d_D, nproj);
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, nband, nproj, one,
              kp_gpu.d_P.data(), npw, d_D, nproj, one, d_hpsi, npw);
  }

  // 4. Low-rank ACE Hybrid Exact Exchange via pure cublasZgemm (ZERO 3D FFTs!)
  if (ace_nocc > 0 && std::abs(alpha_x) > 0.0 && kp_gpu.d_ace_xi.size() >= static_cast<std::size_t>(npw) * ace_nocc) {
    d_proj_work.resize(static_cast<std::size_t>(ace_nocc) * nband);
    ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, ace_nocc, nband, npw, one,
              kp_gpu.d_ace_xi.data(), npw, d_psi, npw, zero, d_proj_work.data(), ace_nocc);
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, nband, ace_nocc,
              make_cuDoubleComplex(-alpha_x, 0.0),
              kp_gpu.d_ace_xi.data(), npw, d_proj_work.data(), ace_nocc, one, d_hpsi, npw);
  }
}

using HpsiCallbackGpu = std::function<void(const cuDoubleComplex* d_psi, cuDoubleComplex* d_hpsi, int nband)>;

void build_initial_guess_gpu(PwCudaContext& ctx,
                             const KPointBasis& kp,
                             int n_solve,
                             const HpsiCallbackGpu& apply_h_gpu,
                             DeviceBuffer<cuDoubleComplex>& d_psi_out,
                             int& hpsi_calls) {
  const int npw = kp.npw;
  const int n_init = std::min(npw, std::max(n_solve * 3, std::min(npw, 64)));

  std::vector<int> order(static_cast<std::size_t>(npw));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    if (std::abs(kp.kin[a] - kp.kin[b]) > 1e-12) return kp.kin[a] < kp.kin[b];
    return kp.fft_indices[a] < kp.fft_indices[b];
  });

  std::vector<Complex> h_V0(static_cast<std::size_t>(npw) * n_init, Complex(0.0, 0.0));
  for (int j = 0; j < n_init; ++j) {
    h_V0[order[j] + static_cast<std::size_t>(j) * npw] = Complex(1.0, 0.0);
  }

  DeviceBuffer<cuDoubleComplex> d_V0;
  d_V0.upload(reinterpret_cast<const cuDoubleComplex*>(h_V0.data()), h_V0.size());
  DeviceBuffer<cuDoubleComplex> d_HV0(static_cast<std::size_t>(npw) * n_init);
  apply_h_gpu(d_V0.data(), d_HV0.data(), n_init);
  hpsi_calls += n_init;

  DeviceBuffer<cuDoubleComplex> d_Hsub(static_cast<std::size_t>(n_init) * n_init);
  ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, n_init, n_init, npw,
            make_cuDoubleComplex(1.0, 0.0), d_V0.data(), npw, d_HV0.data(), npw,
            make_cuDoubleComplex(0.0, 0.0), d_Hsub.data(), n_init);

  DeviceBuffer<double> d_w(static_cast<std::size_t>(n_init));
  zheevd_gpu(ctx, n_init, d_Hsub.data(), d_w.data());

  d_psi_out.resize(static_cast<std::size_t>(npw) * n_solve);
  ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, n_init,
            make_cuDoubleComplex(1.0, 0.0), d_V0.data(), npw, d_Hsub.data(), n_init,
            make_cuDoubleComplex(0.0, 0.0), d_psi_out.data(), npw);
}

// GPU-Resident Block-Davidson & LOBPCG Iterative Eigensolvers
PwEigenResult run_eigensolver_gpu(PwCudaContext& ctx,
                                  const KPointBasis& kp,
                                  const double* d_kin,
                                  int nband_target,
                                  int n_solve,
                                  PwSolverType solver,
                                  double residual_tol,
                                  int max_iter,
                                  const HpsiCallbackGpu& apply_h_gpu,
                                  DeviceBuffer<cuDoubleComplex>& d_psi_inout,
                                  bool download_eigenvectors) {
  const int npw = kp.npw;
  n_solve = std::min(npw, std::max(nband_target, n_solve));
  PwEigenResult res;
  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

  if (npw <= n_solve) {
    std::vector<Complex> h_I(static_cast<std::size_t>(npw) * npw, Complex(0.0, 0.0));
    for (int i = 0; i < npw; ++i) h_I[i + static_cast<std::size_t>(i) * npw] = Complex(1.0, 0.0);
    DeviceBuffer<cuDoubleComplex> d_I;
    d_I.upload(reinterpret_cast<const cuDoubleComplex*>(h_I.data()), h_I.size());
    DeviceBuffer<cuDoubleComplex> d_H(static_cast<std::size_t>(npw) * npw);
    apply_h_gpu(d_I.data(), d_H.data(), npw);
    res.hpsi_calls += npw;
    res.iterations = 1;
    DeviceBuffer<double> d_w(static_cast<std::size_t>(npw));
    zheevd_gpu(ctx, npw, d_H.data(), d_w.data());
    d_psi_inout = std::move(d_H);
    res.eigenvalues.resize(static_cast<std::size_t>(nband_target));
    d_w.download(res.eigenvalues.data(), static_cast<std::size_t>(nband_target));
    if (download_eigenvectors) {
      res.eigenvectors.resize(static_cast<std::size_t>(npw) * nband_target);
      check_cuda(cudaMemcpy(res.eigenvectors.data(), d_psi_inout.data(),
                            res.eigenvectors.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
                 "download exact evecs");
    }
    return res;
  }

  if (d_psi_inout.size() != static_cast<std::size_t>(npw) * n_solve) {
    build_initial_guess_gpu(ctx, kp, n_solve, apply_h_gpu, d_psi_inout, res.hpsi_calls);
  } else {
    const int kept = svqb_orthonormalize_gpu(ctx, npw, d_psi_inout.data(), n_solve, nullptr, 0);
    if (kept < n_solve) {
      build_initial_guess_gpu(ctx, kp, n_solve, apply_h_gpu, d_psi_inout, res.hpsi_calls);
    }
  }

  DeviceBuffer<cuDoubleComplex> d_R_prec(static_cast<std::size_t>(npw) * n_solve);
  DeviceBuffer<double> d_res_norms(static_cast<std::size_t>(n_solve));
  std::vector<double> h_res_norms(static_cast<std::size_t>(n_solve));

  if (solver == PwSolverType::Davidson) {
    const int max_sub = std::min(npw, std::max(2 * n_solve + 4, 4 * n_solve));
    DeviceBuffer<cuDoubleComplex> d_Vsub(static_cast<std::size_t>(npw) * max_sub);
    DeviceBuffer<cuDoubleComplex> d_HVsub(static_cast<std::size_t>(npw) * max_sub);
    check_cuda(cudaMemcpy(d_Vsub.data(), d_psi_inout.data(),
                          static_cast<std::size_t>(npw) * n_solve * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToDevice),
               "init Vsub");
    apply_h_gpu(d_Vsub.data(), d_HVsub.data(), n_solve);
    res.hpsi_calls += n_solve;
    int nsub = n_solve;

    DeviceBuffer<cuDoubleComplex> d_Psi(static_cast<std::size_t>(npw) * n_solve);
    DeviceBuffer<cuDoubleComplex> d_HPsi(static_cast<std::size_t>(npw) * n_solve);
    DeviceBuffer<cuDoubleComplex> d_Hproj(static_cast<std::size_t>(max_sub) * max_sub);
    DeviceBuffer<double> d_wsub(static_cast<std::size_t>(max_sub));
    DeviceBuffer<cuDoubleComplex> d_Wcand(static_cast<std::size_t>(npw) * n_solve);

    for (int iter = 0; iter < max_iter; ++iter) {
      res.iterations = iter + 1;
      ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, nsub, nsub, npw, one,
                d_Vsub.data(), npw, d_HVsub.data(), npw, zero, d_Hproj.data(), nsub);
      zheevd_gpu(ctx, nsub, d_Hproj.data(), d_wsub.data());

      ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, nsub, one,
                d_Vsub.data(), npw, d_Hproj.data(), nsub, zero, d_Psi.data(), npw);
      ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, nsub, one,
                d_HVsub.data(), npw, d_Hproj.data(), nsub, zero, d_HPsi.data(), npw);

      pw_band_residual_precond_kernel<<<n_solve, 256>>>(
          npw, d_kin, d_wsub.data(), d_Psi.data(), d_HPsi.data(),
          d_R_prec.data(), d_res_norms.data());
      d_res_norms.download(h_res_norms.data(), static_cast<std::size_t>(n_solve));

      double max_res = 0.0;
      std::vector<int> unconv_cols;
      for (int b = 0; b < n_solve; ++b) {
        if (b < nband_target) max_res = std::max(max_res, h_res_norms[b]);
        const double b_tol = (b < nband_target) ? residual_tol : (10.0 * residual_tol);
        if (h_res_norms[b] > b_tol) unconv_cols.push_back(b);
      }
      res.max_residual = max_res;
      if (max_res < residual_tol || unconv_cols.empty() || iter + 1 == max_iter) {
        res.eigenvalues.resize(static_cast<std::size_t>(nband_target));
        d_wsub.download(res.eigenvalues.data(), static_cast<std::size_t>(nband_target));
        break;
      }

      const int n_unconv = static_cast<int>(unconv_cols.size());
      {
        DeviceBuffer<int> d_unconv;
        d_unconv.upload(unconv_cols.data(), unconv_cols.size());
        pw_gather_cols_kernel<<<(npw * n_unconv + 255) / 256, 256>>>(
            npw, n_unconv, d_unconv.data(), d_R_prec.data(), d_Wcand.data());
      }

      if (nsub + n_unconv > max_sub) {
        check_cuda(cudaMemcpy(d_Vsub.data(), d_Psi.data(),
                              static_cast<std::size_t>(npw) * n_solve * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToDevice),
                   "restart Vsub");
        check_cuda(cudaMemcpy(d_HVsub.data(), d_HPsi.data(),
                              static_cast<std::size_t>(npw) * n_solve * sizeof(cuDoubleComplex),
                              cudaMemcpyDeviceToDevice),
                   "restart HVsub");
        nsub = n_solve;
      }

      const int n_add = svqb_orthonormalize_gpu(
          ctx, npw, d_Wcand.data(), n_unconv, d_Vsub.data(), nsub);
      if (n_add == 0) {
        res.eigenvalues.resize(static_cast<std::size_t>(nband_target));
        d_wsub.download(res.eigenvalues.data(), static_cast<std::size_t>(nband_target));
        break;
      }

      cuDoubleComplex* d_Vnext = d_Vsub.data() + static_cast<std::size_t>(nsub) * npw;
      cuDoubleComplex* d_HVnext = d_HVsub.data() + static_cast<std::size_t>(nsub) * npw;
      check_cuda(cudaMemcpy(d_Vnext, d_Wcand.data(),
                            static_cast<std::size_t>(npw) * n_add * sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToDevice),
                 "append Vnext");
      apply_h_gpu(d_Vnext, d_HVnext, n_add);
      res.hpsi_calls += n_add;
      nsub += n_add;

      if (iter + 1 == max_iter) {
        res.eigenvalues.resize(static_cast<std::size_t>(nband_target));
        d_wsub.download(res.eigenvalues.data(), static_cast<std::size_t>(nband_target));
      }
    }

    d_psi_inout = std::move(d_Psi);
    if (download_eigenvectors) {
      res.eigenvectors.resize(static_cast<std::size_t>(npw) * nband_target);
      check_cuda(cudaMemcpy(res.eigenvectors.data(), d_psi_inout.data(),
                            res.eigenvectors.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
                 "download davidson evecs");
    }
    return res;
  }

  // GPU Hetmaniuk-Lehoucq Orthonormalized LOBPCG
  DeviceBuffer<cuDoubleComplex> d_X(static_cast<std::size_t>(npw) * n_solve);
  DeviceBuffer<cuDoubleComplex> d_HX(static_cast<std::size_t>(npw) * n_solve);
  check_cuda(cudaMemcpy(d_X.data(), d_psi_inout.data(),
                        static_cast<std::size_t>(npw) * n_solve * sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToDevice),
             "init LOBPCG X");
  apply_h_gpu(d_X.data(), d_HX.data(), n_solve);
  res.hpsi_calls += n_solve;

  DeviceBuffer<double> d_lambda(static_cast<std::size_t>(3 * n_solve));
  {
    DeviceBuffer<cuDoubleComplex> d_Hx(static_cast<std::size_t>(n_solve) * n_solve);
    ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, n_solve, n_solve, npw, one,
              d_X.data(), npw, d_HX.data(), npw, zero, d_Hx.data(), n_solve);
    zheevd_gpu(ctx, n_solve, d_Hx.data(), d_lambda.data());
    DeviceBuffer<cuDoubleComplex> d_Xrot(static_cast<std::size_t>(npw) * n_solve);
    DeviceBuffer<cuDoubleComplex> d_HXrot(static_cast<std::size_t>(npw) * n_solve);
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, n_solve, one,
              d_X.data(), npw, d_Hx.data(), n_solve, zero, d_Xrot.data(), npw);
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, n_solve, one,
              d_HX.data(), npw, d_Hx.data(), n_solve, zero, d_HXrot.data(), npw);
    d_X = std::move(d_Xrot);
    d_HX = std::move(d_HXrot);
  }

  DeviceBuffer<cuDoubleComplex> d_P(static_cast<std::size_t>(npw) * n_solve);
  DeviceBuffer<cuDoubleComplex> d_HP(static_cast<std::size_t>(npw) * n_solve);
  DeviceBuffer<cuDoubleComplex> d_W(static_cast<std::size_t>(npw) * n_solve);
  DeviceBuffer<cuDoubleComplex> d_HW(static_cast<std::size_t>(npw) * n_solve);
  DeviceBuffer<cuDoubleComplex> d_V(static_cast<std::size_t>(npw) * (3 * n_solve));
  DeviceBuffer<cuDoubleComplex> d_HV(static_cast<std::size_t>(npw) * (3 * n_solve));
  DeviceBuffer<cuDoubleComplex> d_Hsub(static_cast<std::size_t>(3 * n_solve) * (3 * n_solve));
  int n_p = 0;

  for (int iter = 0; iter < max_iter; ++iter) {
    res.iterations = iter + 1;
    pw_band_residual_precond_kernel<<<n_solve, 256>>>(
        npw, d_kin, d_lambda.data(), d_X.data(), d_HX.data(),
        d_R_prec.data(), d_res_norms.data());
    d_res_norms.download(h_res_norms.data(), static_cast<std::size_t>(n_solve));

    double max_res = 0.0;
    std::vector<int> unconv_cols;
    for (int b = 0; b < n_solve; ++b) {
      if (b < nband_target) max_res = std::max(max_res, h_res_norms[b]);
      const double b_tol = (b < nband_target) ? residual_tol : (10.0 * residual_tol);
      if (h_res_norms[b] > b_tol) unconv_cols.push_back(b);
    }
    res.max_residual = max_res;
    if (max_res < residual_tol || unconv_cols.empty()) break;

    int n_w = static_cast<int>(unconv_cols.size());
    {
      DeviceBuffer<int> d_unconv;
      d_unconv.upload(unconv_cols.data(), unconv_cols.size());
      pw_gather_cols_kernel<<<(npw * n_w + 255) / 256, 256>>>(
          npw, n_w, d_unconv.data(), d_R_prec.data(), d_W.data());
    }

    n_w = svqb_orthonormalize_gpu(ctx, npw, d_W.data(), n_w, d_X.data(), n_solve);
    if (n_w == 0) break;

    apply_h_gpu(d_W.data(), d_HW.data(), n_w);
    res.hpsi_calls += n_w;

    const int n_xw = n_solve + n_w;
    check_cuda(cudaMemcpy(d_V.data(), d_X.data(),
                          static_cast<std::size_t>(npw) * n_solve * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToDevice),
               "V <- X");
    check_cuda(cudaMemcpy(d_HV.data(), d_HX.data(),
                          static_cast<std::size_t>(npw) * n_solve * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToDevice),
               "HV <- HX");
    check_cuda(cudaMemcpy(d_V.data() + static_cast<std::size_t>(npw) * n_solve, d_W.data(),
                          static_cast<std::size_t>(npw) * n_w * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToDevice),
               "V <- W");
    check_cuda(cudaMemcpy(d_HV.data() + static_cast<std::size_t>(npw) * n_solve, d_HW.data(),
                          static_cast<std::size_t>(npw) * n_w * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToDevice),
               "HV <- HW");

    int n_p_act = 0;
    if (n_p > 0 && n_xw < npw) {
      n_p_act = std::min(n_p, npw - n_xw);
      cuDoubleComplex* d_Vp = d_V.data() + static_cast<std::size_t>(npw) * n_xw;
      cuDoubleComplex* d_HVp = d_HV.data() + static_cast<std::size_t>(npw) * n_xw;
      check_cuda(cudaMemcpy(d_Vp, d_P.data(),
                            static_cast<std::size_t>(npw) * n_p_act * sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToDevice),
                 "V <- P");
      check_cuda(cudaMemcpy(d_HVp, d_HP.data(),
                            static_cast<std::size_t>(npw) * n_p_act * sizeof(cuDoubleComplex),
                            cudaMemcpyDeviceToDevice),
                 "HV <- HP");
      n_p_act = svqb_orthonormalize_gpu(
          ctx, npw, d_Vp, n_p_act, d_V.data(), n_xw, d_HVp, d_HV.data());
    }

    const int nsub = n_xw + n_p_act;
    ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, nsub, nsub, npw, one,
              d_V.data(), npw, d_HV.data(), npw, zero, d_Hsub.data(), nsub);
    zheevd_gpu(ctx, nsub, d_Hsub.data(), d_lambda.data());

    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, nsub, one,
              d_V.data(), npw, d_Hsub.data(), nsub, zero, d_X.data(), npw);
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, nsub, one,
              d_HV.data(), npw, d_Hsub.data(), nsub, zero, d_HX.data(), npw);

    const int n_wp = nsub - n_solve;
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, n_wp, one,
              d_V.data() + static_cast<std::size_t>(npw) * n_solve, npw,
              d_Hsub.data() + n_solve, nsub, zero, d_P.data(), npw);
    ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, npw, n_solve, n_wp, one,
              d_HV.data() + static_cast<std::size_t>(npw) * n_solve, npw,
              d_Hsub.data() + n_solve, nsub, zero, d_HP.data(), npw);
    n_p = n_solve;
  }

  res.eigenvalues.resize(static_cast<std::size_t>(nband_target));
  d_lambda.download(res.eigenvalues.data(), static_cast<std::size_t>(nband_target));
  d_psi_inout = std::move(d_X);
  if (download_eigenvectors) {
    res.eigenvectors.resize(static_cast<std::size_t>(npw) * nband_target);
    check_cuda(cudaMemcpy(res.eigenvectors.data(), d_psi_inout.data(),
                          res.eigenvectors.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
               "download lobpcg evecs");
  }
  return res;
}

// GPU Exact Exchange Action via Batched 3D Poisson cuFFTs (`cufftPlanMany`)
void apply_exact_exchange_poisson_gpu(PwCudaContext& ctx,
                                      const PwBasis& basis,
                                      int ik_target,
                                      int n_occ,
                                      const std::vector<GpuKPointData>& kp_gpu_all,
                                      const std::vector<DeviceBuffer<cuDoubleComplex>>& d_u_occ_r_all_k,
                                      const std::vector<DeviceBuffer<double>>& d_coul_table,
                                      const cuDoubleComplex* d_psi_trial,
                                      int n_trial,
                                      bool symmetric_occ_square,
                                      cuDoubleComplex* d_vx_psi_out,
                                      int& poisson_fft_counter) {
  const int ngrid = basis.ngrid();
  const int nkpts = basis.nkpts();
  const double vol = basis.cell().volume();
  const double inv_sqrt_vol = 1.0 / std::sqrt(vol);
  const double inv_ngrid = 1.0 / static_cast<double>(ngrid);
  const auto& kp_t = kp_gpu_all[ik_target];
  const int npw = kp_t.npw;
  const int threads = 256;

  DeviceBuffer<cuDoubleComplex> d_u_trial_r;
  const cuDoubleComplex* d_u_trial_ptr = nullptr;
  if (symmetric_occ_square && n_trial == n_occ) {
    d_u_trial_ptr = d_u_occ_r_all_k[ik_target].data();
  } else {
    d_u_trial_r.resize(static_cast<std::size_t>(n_trial) * ngrid);
    d_u_trial_r.zero();
    pw_scatter_sphere_to_box_kernel<<<(npw * n_trial + threads - 1) / threads, threads>>>(
        npw, n_trial, ngrid, kp_t.d_fft_idx.data(), d_psi_trial, d_u_trial_r.data());
    ctx.exec_fft3d(n_trial, d_u_trial_r.data(), d_u_trial_r.data(), +1);
    pw_scale_complex_kernel<<<static_cast<unsigned int>((d_u_trial_r.size() + threads - 1) / threads), threads>>>(
        d_u_trial_r.size(), inv_sqrt_vol, d_u_trial_r.data());
    d_u_trial_ptr = d_u_trial_r.data();
  }

  DeviceBuffer<cuDoubleComplex> d_w_r(static_cast<std::size_t>(n_trial) * ngrid);
  d_w_r.zero();

  const int max_i_chunk = std::max(1, 64 / std::max(1, n_trial));
  const int symm_pairs = (symmetric_occ_square && n_trial == n_occ && n_occ <= 16)
                             ? (n_occ * (n_occ + 1) / 2)
                             : 0;
  const int pair_buf_grids = std::max(max_i_chunk * n_trial, symm_pairs);
  DeviceBuffer<cuDoubleComplex> d_pair_buf(static_cast<std::size_t>(pair_buf_grids) * ngrid);

  for (int iq = 0; iq < nkpts; ++iq) {
    const double wq = basis.kpoint(iq).weight;
    const double* d_coul_ptr = d_coul_table[static_cast<std::size_t>(ik_target * nkpts + iq)].data();

    if (symm_pairs > 0 && iq == ik_target) {
      const std::size_t tot_pair_pts = static_cast<std::size_t>(symm_pairs) * ngrid;
      pw_form_symm_pairs_kernel<<<static_cast<unsigned int>((tot_pair_pts + threads - 1) / threads), threads>>>(
          ngrid, n_occ, d_u_occ_r_all_k[iq].data(), d_pair_buf.data());

      ctx.exec_fft3d(symm_pairs, d_pair_buf.data(), d_pair_buf.data(), -1);
      pw_mul_coulomb_kernel<<<static_cast<unsigned int>((tot_pair_pts + threads - 1) / threads), threads>>>(
          ngrid, symm_pairs, inv_ngrid, d_coul_ptr, d_pair_buf.data());
      ctx.exec_fft3d(symm_pairs, d_pair_buf.data(), d_pair_buf.data(), +1);
      poisson_fft_counter += 2 * symm_pairs;

      const std::size_t tot_w_pts = static_cast<std::size_t>(n_occ) * ngrid;
      pw_accum_symm_exx_kernel<<<static_cast<unsigned int>((tot_w_pts + threads - 1) / threads), threads>>>(
          ngrid, n_occ, wq, d_u_occ_r_all_k[iq].data(), d_pair_buf.data(), d_w_r.data());
      continue;
    }

    for (int i0 = 0; i0 < n_occ; i0 += max_i_chunk) {
      const int cur_i = std::min(max_i_chunk, n_occ - i0);
      const int npairs = cur_i * n_trial;
      const std::size_t tot_pair_pts = static_cast<std::size_t>(npairs) * ngrid;

      pw_form_rect_pairs_kernel<<<static_cast<unsigned int>((tot_pair_pts + threads - 1) / threads), threads>>>(
          ngrid, i0, cur_i, n_trial, d_u_occ_r_all_k[iq].data(),
          d_u_trial_ptr, d_pair_buf.data());

      ctx.exec_fft3d(npairs, d_pair_buf.data(), d_pair_buf.data(), -1);
      pw_mul_coulomb_kernel<<<static_cast<unsigned int>((tot_pair_pts + threads - 1) / threads), threads>>>(
          ngrid, npairs, inv_ngrid, d_coul_ptr, d_pair_buf.data());
      ctx.exec_fft3d(npairs, d_pair_buf.data(), d_pair_buf.data(), +1);
      poisson_fft_counter += 2 * npairs;

      const std::size_t tot_w_pts = static_cast<std::size_t>(n_trial) * ngrid;
      pw_accum_rect_exx_kernel<<<static_cast<unsigned int>((tot_w_pts + threads - 1) / threads), threads>>>(
          ngrid, i0, cur_i, n_trial, wq, d_u_occ_r_all_k[iq].data(),
          d_pair_buf.data(), d_w_r.data());
    }
  }

  ctx.exec_fft3d(n_trial, d_w_r.data(), d_w_r.data(), -1);
  const double scale_back = std::sqrt(vol) * inv_ngrid;
  pw_gather_scaled_box_to_sphere_kernel<<<(npw * n_trial + threads - 1) / threads, threads>>>(
      npw, n_trial, ngrid, scale_back, kp_t.d_fft_idx.data(), d_w_r.data(), d_vx_psi_out);
}

// Kerker-preconditioned Pulay DIIS helper (operates on compact G-space density history)
class GpuHostKerkerDiis {
 public:
  GpuHostKerkerDiis(int ngrid, const std::vector<double>& g2, double vol,
                    double beta, double q0, int max_vec)
      : ngrid_(ngrid), g2_(g2), vol_(vol), beta_(beta), q0_sq_(q0 * q0), max_vec_(std::max(1, max_vec)) {}

  double step(PwCudaContext& ctx,
              double n_elec,
              const Complex* h_rho_in_g,
              const Complex* h_rho_out_g,
              DeviceBuffer<cuDoubleComplex>& d_buf_c,
              DeviceBuffer<double>& d_rho_r) {
    std::vector<Complex> res_g(static_cast<std::size_t>(ngrid_));
    double rms2 = 0.0;
    for (int ig = 0; ig < ngrid_; ++ig) {
      res_g[ig] = h_rho_out_g[ig] - h_rho_in_g[ig];
      rms2 += std::norm(res_g[ig]);
    }
    res_g[0] = Complex(0.0, 0.0);
    const double rms = std::sqrt(rms2 * vol_);

    if (static_cast<int>(rho_in_hist_.size()) >= max_vec_) {
      rho_in_hist_.erase(rho_in_hist_.begin());
      res_hist_.erase(res_hist_.begin());
    }
    rho_in_hist_.emplace_back(h_rho_in_g, h_rho_in_g + ngrid_);
    res_hist_.push_back(std::move(res_g));

    const int m = static_cast<int>(rho_in_hist_.size());
    std::vector<double> weights(static_cast<std::size_t>(m), 0.0);
    weights[m - 1] = 1.0;

    if (m >= 2) {
      const int dim = m + 1;
      std::vector<double> A(static_cast<std::size_t>(dim) * dim, 0.0);
      std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
      double max_diag = 0.0;
      for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
          double dot = 0.0;
          for (int ig = 1; ig < ngrid_; ++ig) {
            const double w_k = (g2_[ig] + q0_sq_) / std::max(g2_[ig], 1e-12);
            dot += w_k * (res_hist_[i][ig].real() * res_hist_[j][ig].real() +
                          res_hist_[i][ig].imag() * res_hist_[j][ig].imag());
          }
          dot *= vol_;
          A[i * dim + j] = dot;
          A[j * dim + i] = dot;
          if (i == j) max_diag = std::max(max_diag, dot);
        }
        A[i * dim + m] = 1.0;
        A[m * dim + i] = 1.0;
      }
      rhs[m] = 1.0;
      const double reg = std::max(1e-28, 1e-13 * max_diag);
      for (int i = 0; i < m; ++i) A[i * dim + i] += reg;

      bool ok = true;
      for (int k = 0; k < dim; ++k) {
        int piv = k;
        double mv = std::abs(A[k * dim + k]);
        for (int i = k + 1; i < dim; ++i) {
          if (std::abs(A[i * dim + k]) > mv) {
            mv = std::abs(A[i * dim + k]);
            piv = i;
          }
        }
        if (mv < 1e-30) {
          ok = false;
          break;
        }
        if (piv != k) {
          for (int j = k; j < dim; ++j) std::swap(A[k * dim + j], A[piv * dim + j]);
          std::swap(rhs[k], rhs[piv]);
        }
        for (int i = k + 1; i < dim; ++i) {
          const double f = A[i * dim + k] / A[k * dim + k];
          for (int j = k; j < dim; ++j) A[i * dim + j] -= f * A[k * dim + j];
          rhs[i] -= f * rhs[k];
        }
      }
      if (ok) {
        std::vector<double> sol(static_cast<std::size_t>(dim), 0.0);
        for (int i = dim - 1; i >= 0; --i) {
          double s = rhs[i];
          for (int j = i + 1; j < dim; ++j) s -= A[i * dim + j] * sol[j];
          sol[i] = s / A[i * dim + i];
        }
        double mw = 0.0;
        for (int i = 0; i < m; ++i) mw = std::max(mw, std::abs(sol[i]));
        if (std::isfinite(mw) && mw < 20.0) {
          for (int i = 0; i < m; ++i) weights[i] = sol[i];
        }
      }
    }

    std::vector<Complex> mixed_g(static_cast<std::size_t>(ngrid_), Complex(0.0, 0.0));
    mixed_g[0] = Complex(n_elec / vol_, 0.0);
    for (int ig = 1; ig < ngrid_; ++ig) {
      Complex r_opt(0.0, 0.0), res_opt(0.0, 0.0);
      for (int a = 0; a < m; ++a) {
        r_opt += weights[a] * rho_in_hist_[a][ig];
        res_opt += weights[a] * res_hist_[a][ig];
      }
      mixed_g[ig] = r_opt + (beta_ * g2_[ig] / (g2_[ig] + q0_sq_)) * res_opt;
    }

    d_buf_c.upload(reinterpret_cast<const cuDoubleComplex*>(mixed_g.data()), mixed_g.size());
    ctx.exec_fft3d(1, d_buf_c.data(), d_buf_c.data(), +1);
    pw_complex_to_real_kernel<<<(ngrid_ + 255) / 256, 256>>>(
        ngrid_, d_buf_c.data(), d_rho_r.data());
    return rms;
  }

 private:
  int ngrid_ = 0;
  const std::vector<double>& g2_;
  double vol_ = 1.0;
  double beta_ = 0.6;
  double q0_sq_ = 0.64;
  int max_vec_ = 8;
  std::vector<std::vector<Complex>> rho_in_hist_;
  std::vector<std::vector<Complex>> res_hist_;
};

}  // namespace

// ============================================================================
// Public Detail Entry Points Linked by `src/pw.cpp`
// ============================================================================

int pw_cuda_device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
  return count;
}

void pw_cuda_fft3d_batch(std::array<int, 3> grid,
                         int nbatch,
                         const Complex* h_in,
                         Complex* h_out,
                         int direction) {
  if (nbatch <= 0) return;
  PwCudaContext ctx(grid);
  const std::size_t total = static_cast<std::size_t>(nbatch) * static_cast<std::size_t>(ctx.ngrid());
  DeviceBuffer<cuDoubleComplex> d_buf;
  d_buf.upload(reinterpret_cast<const cuDoubleComplex*>(h_in), total);
  ctx.exec_fft3d(nbatch, d_buf.data(), d_buf.data(), direction);
  d_buf.download(reinterpret_cast<cuDoubleComplex*>(h_out), total);
}

PwEigenResult pw_cuda_solve_eigenproblem(const PwBasis& basis,
                                         int ik,
                                         std::span<const double> vloc_r,
                                         int nband,
                                         PwSolverType solver,
                                         double residual_tol,
                                         int max_iter,
                                         bool include_nonlocal_pp) {
  PwCudaContext ctx(basis.fft_grid());
  GpuKPointData kp_gpu;
  upload_kpoint_data(basis.cell(), basis.kpoint(ik), include_nonlocal_pp, kp_gpu);

  DeviceBuffer<double> d_vloc;
  d_vloc.upload(vloc_r.data(), vloc_r.size());

  DeviceBuffer<cuDoubleComplex> d_fft_work;
  DeviceBuffer<cuDoubleComplex> d_proj_work;
  DeviceBuffer<cuDoubleComplex> d_psi;

  const int n_solve = std::min(basis.kpoint(ik).npw, std::max(nband + 4, nband * 2));
  auto apply_h = [&](const cuDoubleComplex* d_x, cuDoubleComplex* d_hx, int nb) {
    apply_pw_hamiltonian_gpu(ctx, kp_gpu, d_vloc.data(), 0, 0.0,
                             d_x, d_hx, nb, d_fft_work, d_proj_work);
  };
  return run_eigensolver_gpu(ctx, basis.kpoint(ik), kp_gpu.d_kin.data(),
                             nband, n_solve, solver, residual_tol, max_iter,
                             apply_h, d_psi, true);
}

PwScfResult pw_cuda_run_scf(const PwBasis& basis, const PwScfOptions& opts) {
  const auto t_start = std::chrono::steady_clock::now();
  const int n_elec = basis.cell().num_electrons(opts.charge);
  if (n_elec % 2 != 0) {
    throw std::invalid_argument("PwKohnShamSolver (CUDA): closed-shell RKS requires even electron count");
  }
  const int n_occ = n_elec / 2;
  const int nkpts = basis.nkpts();
  const int ngrid = basis.ngrid();
  const double vol = basis.cell().volume();
  const double dvol = vol / static_cast<double>(ngrid);
  const double inv_ngrid = 1.0 / static_cast<double>(ngrid);
  const int threads = 256;
  const int grid_blocks = (ngrid + threads - 1) / threads;

  const auto xc_id = xc_gen::parse_xc_functional(opts.xc);
  const double alpha_x = xc_gen::xc_exact_exchange_fraction_by_id(xc_id);
  const bool is_hybrid = (std::abs(alpha_x) > 0.0);

  int min_npw = basis.kpoint(0).npw;
  for (int ik = 1; ik < nkpts; ++ik) min_npw = std::min(min_npw, basis.kpoint(ik).npw);

  const int nband_out =
      (opts.nbands > 0) ? std::min(min_npw, std::max(opts.nbands, n_occ))
                        : std::min(min_npw, std::max(n_occ + 2, n_occ));
  const int n_solve = std::min(min_npw, std::max(nband_out + 4, n_occ + 6));

  PwScfResult res;
  res.nbands = nband_out;
  res.ewald_energy = basis.cell().ewald_energy();

  PwCudaContext ctx(basis.fft_grid());

  // Upload static grid & pseudopotential buffers to GPU once
  const std::vector<double> h_vloc_ion_r = basis.compute_local_pp_grid();
  DeviceBuffer<double> d_vloc_ion_r;
  d_vloc_ion_r.upload(h_vloc_ion_r.data(), h_vloc_ion_r.size());

  DeviceBuffer<double> d_g2;
  d_g2.upload(basis.g2().data(), basis.g2().size());
  DeviceBuffer<double> d_g_xyz;
  d_g_xyz.upload(basis.g_vecs_xyz().data(), basis.g_vecs_xyz().size());

  std::vector<GpuKPointData> kp_gpu(static_cast<std::size_t>(nkpts));
  for (int ik = 0; ik < nkpts; ++ik) {
    upload_kpoint_data(basis.cell(), basis.kpoint(ik), true, kp_gpu[ik]);
  }

  // Initial superposition-of-atomic-Gaussians density on GPU
  DeviceBuffer<double> d_rho_r(static_cast<std::size_t>(ngrid));
  DeviceBuffer<cuDoubleComplex> d_buf_c(static_cast<std::size_t>(ngrid));
  {
    const double z_tot = basis.cell().total_valence_charge();
    const double scale_q = (z_tot > 0.0) ? (static_cast<double>(n_elec) / z_tot) : 0.0;
    std::vector<Complex> h_rho_g(static_cast<std::size_t>(ngrid), Complex(0.0, 0.0));
    const auto& gv = basis.g_vecs_xyz();
    const auto& g2 = basis.g2();
    if (basis.cell().natom() == 0 || z_tot <= 0.0) {
      h_rho_g[0] = Complex(static_cast<double>(n_elec) / vol, 0.0);
    } else {
      for (const auto& atm : basis.cell().atoms()) {
        const double q_atm = scale_q * atm.pseudo.z_ion;
        const double sigma = std::max(0.75, 1.8 * atm.pseudo.r_loc);
        const double half_s2 = 0.5 * sigma * sigma;
        for (int ig = 0; ig < ngrid; ++ig) {
          const double phase = -(gv[3 * ig + 0] * atm.pos[0] +
                                 gv[3 * ig + 1] * atm.pos[1] +
                                 gv[3 * ig + 2] * atm.pos[2]);
          const double env = (q_atm / vol) * std::exp(-half_s2 * g2[ig]);
          h_rho_g[ig] += env * Complex(std::cos(phase), std::sin(phase));
        }
      }
      h_rho_g[0] = Complex(static_cast<double>(n_elec) / vol, 0.0);
    }
    d_buf_c.upload(reinterpret_cast<const cuDoubleComplex*>(h_rho_g.data()), h_rho_g.size());
    ctx.exec_fft3d(1, d_buf_c.data(), d_buf_c.data(), +1);
    std::vector<Complex> h_rho_r_c(static_cast<std::size_t>(ngrid));
    d_buf_c.download(reinterpret_cast<cuDoubleComplex*>(h_rho_r_c.data()), h_rho_r_c.size());
    std::vector<double> h_rho_init(static_cast<std::size_t>(ngrid));
    for (int ir = 0; ir < ngrid; ++ir) h_rho_init[ir] = std::max(1e-14, h_rho_r_c[ir].real());
    d_rho_r.upload(h_rho_init.data(), h_rho_init.size());
  }

  // Persistent GPU workspaces
  DeviceBuffer<cuDoubleComplex> d_rho_g(static_cast<std::size_t>(ngrid));
  DeviceBuffer<double> d_vh_r(static_cast<std::size_t>(ngrid));
  DeviceBuffer<double> d_vxc_r(static_cast<std::size_t>(ngrid));
  DeviceBuffer<double> d_veff_r(static_cast<std::size_t>(ngrid));
  DeviceBuffer<cuDoubleComplex> d_grad_buf(static_cast<std::size_t>(3) * ngrid);
  DeviceBuffer<cuDoubleComplex> d_fft_work;
  DeviceBuffer<cuDoubleComplex> d_proj_work;
  DeviceBuffer<double> d_scalars(2);  // [eh, exc]

  std::vector<DeviceBuffer<cuDoubleComplex>> d_psi_k(static_cast<std::size_t>(nkpts));
  std::vector<std::vector<double>> evals_k(static_cast<std::size_t>(nkpts));

  // Evaluates V_H(r), V_xc(r), V_eff(r), E_H, and E_xc 100% on GPU
  auto eval_hartree_xc_veff_gpu = [&](const DeviceBuffer<double>& d_rho_in,
                                      xc_gen::XcFunctionalId eval_xc_id,
                                      double& eh_out,
                                      double& exc_out) {
    d_scalars.zero();
    pw_real_to_complex_kernel<<<grid_blocks, threads>>>(ngrid, d_rho_in.data(), d_rho_g.data());
    ctx.exec_fft3d(1, d_rho_g.data(), d_rho_g.data(), -1);
    pw_scale_complex_kernel<<<grid_blocks, threads>>>(
        static_cast<std::size_t>(ngrid), inv_ngrid, d_rho_g.data());

    pw_hartree_g_kernel<<<grid_blocks, threads>>>(
        ngrid, vol, d_g2.data(), d_rho_g.data(), d_buf_c.data(), d_scalars.data());
    ctx.exec_fft3d(1, d_buf_c.data(), d_buf_c.data(), +1);
    pw_complex_to_real_kernel<<<grid_blocks, threads>>>(ngrid, d_buf_c.data(), d_vh_r.data());

    if (eval_xc_id == xc_gen::XcFunctionalId::k_hf) {
      d_vxc_r.zero();
      pw_assemble_veff_kernel<<<grid_blocks, threads>>>(
          ngrid, 0, d_vloc_ion_r.data(), d_vh_r.data(), nullptr, d_vxc_r.data(), d_veff_r.data());
    } else {
      const bool is_gga = (xc_gen::xc_functional_family(eval_xc_id) != xc_gen::XcFamily::LDA);
      if (is_gga) {
        pw_gga_grad_prep_kernel<<<grid_blocks, threads>>>(
            ngrid, d_g_xyz.data(), d_rho_g.data(), d_grad_buf.data());
        ctx.exec_fft3d(3, d_grad_buf.data(), d_grad_buf.data(), +1);
      }
      pw_eval_xc_gpu_kernel<<<grid_blocks, threads>>>(
          ngrid, static_cast<int>(eval_xc_id), is_gga ? 1 : 0, opts.rho_cutoff, dvol,
          d_rho_in.data(), d_grad_buf.data(), d_vxc_r.data(), d_scalars.data() + 1);
      if (is_gga) {
        ctx.exec_fft3d(3, d_grad_buf.data(), d_grad_buf.data(), -1);
        pw_gga_div_prep_kernel<<<grid_blocks, threads>>>(
            ngrid, inv_ngrid, d_g_xyz.data(), d_grad_buf.data(), d_buf_c.data());
        ctx.exec_fft3d(1, d_buf_c.data(), d_buf_c.data(), +1);
      }
      pw_assemble_veff_kernel<<<grid_blocks, threads>>>(
          ngrid, is_gga ? 1 : 0, d_vloc_ion_r.data(), d_vh_r.data(),
          d_buf_c.data(), d_vxc_r.data(), d_veff_r.data());
    }
    double h_sc[2] = {0.0, 0.0};
    d_scalars.download(h_sc, 2);
    eh_out = h_sc[0];
    exc_out = h_sc[1];
  };

  // Builds rho_out(r) on GPU via batched `cufftPlanMany` across all `n_occ` occupied bands!
  auto compute_density_gpu = [&](DeviceBuffer<double>& d_rho_out) {
    d_rho_out.resize(static_cast<std::size_t>(ngrid));
    d_rho_out.zero();
    if (n_occ <= 0) return;
    d_fft_work.resize(static_cast<std::size_t>(n_occ) * ngrid);
    for (int ik = 0; ik < nkpts; ++ik) {
      const auto& kp_g = kp_gpu[ik];
      check_cuda(cudaMemset(d_fft_work.data(), 0,
                            static_cast<std::size_t>(n_occ) * ngrid * sizeof(cuDoubleComplex)),
                 "memset density fft_work");
      pw_scatter_sphere_to_box_kernel<<<(kp_g.npw * n_occ + threads - 1) / threads, threads>>>(
          kp_g.npw, n_occ, ngrid, kp_g.d_fft_idx.data(), d_psi_k[ik].data(), d_fft_work.data());
      ctx.exec_fft3d(n_occ, d_fft_work.data(), d_fft_work.data(), +1);
      const double wk_occ = 2.0 * kp_g.weight / vol;
      pw_accum_density_kernel<<<grid_blocks, threads>>>(
          ngrid, n_occ, wk_occ, d_fft_work.data(), d_rho_out.data());
    }
  };

  // Evaluates exact variational total energy components on GPU
  DeviceBuffer<double> d_ekin_acc(1);
  auto compute_variational_components_gpu = [&](const DeviceBuffer<double>& d_rho_eval,
                                                xc_gen::XcFunctionalId eval_xc_id,
                                                double exx_raw_unscaled) {
    double eh = 0.0, exc = 0.0;
    eval_hartree_xc_veff_gpu(d_rho_eval, eval_xc_id, eh, exc);

    double e_loc = 0.0;
    check_cublas(cublasDdot(ctx.cublas(), ngrid, d_rho_eval.data(), 1, d_vloc_ion_r.data(), 1, &e_loc),
                 "cublasDdot e_loc");
    e_loc *= dvol;

    d_ekin_acc.zero();
    double e_nl = 0.0;
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    for (int ik = 0; ik < nkpts; ++ik) {
      const int npw = kp_gpu[ik].npw;
      const double wk_occ = 2.0 * kp_gpu[ik].weight;
      if (n_occ > 0) {
        pw_ekin_bands_kernel<<<n_occ, 256>>>(
            npw, n_occ, wk_occ, kp_gpu[ik].d_kin.data(), d_psi_k[ik].data(), d_ekin_acc.data());
      }
      if (kp_gpu[ik].nproj > 0 && n_occ > 0) {
        const int nproj = kp_gpu[ik].nproj;
        DeviceBuffer<cuDoubleComplex> d_C(static_cast<std::size_t>(nproj) * n_occ);
        DeviceBuffer<cuDoubleComplex> d_D(static_cast<std::size_t>(nproj) * n_occ);
        ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, nproj, n_occ, npw, one,
                  kp_gpu[ik].d_P.data(), npw, d_psi_k[ik].data(), npw, zero, d_C.data(), nproj);
        ctx.zgemm(CUBLAS_OP_N, CUBLAS_OP_N, nproj, n_occ, nproj, one,
                  kp_gpu[ik].d_H_proj.data(), nproj, d_C.data(), nproj, zero, d_D.data(), nproj);
        std::vector<Complex> h_C(d_C.size()), h_D(d_D.size());
        check_cuda(cudaMemcpy(h_C.data(), d_C.data(), h_C.size() * sizeof(Complex), cudaMemcpyDeviceToHost), "C");
        check_cuda(cudaMemcpy(h_D.data(), d_D.data(), h_D.size() * sizeof(Complex), cudaMemcpyDeviceToHost), "D");
        for (std::size_t idx = 0; idx < h_C.size(); ++idx) {
          e_nl += wk_occ * (std::conj(h_C[idx]) * h_D[idx]).real();
        }
      }
    }
    double e_kin = 0.0;
    d_ekin_acc.download(&e_kin, 1);

    res.kinetic_energy = e_kin;
    res.local_pp_energy = e_loc;
    res.nonlocal_pp_energy = e_nl;
    res.hartree_energy = eh;
    res.xc_energy = exc;
    res.exx_energy = alpha_x * exx_raw_unscaled;
    res.total_energy = res.ewald_energy + e_kin + e_loc + e_nl + eh + exc + res.exx_energy;
  };

  auto run_semilocal_or_ace_loop_gpu = [&](xc_gen::XcFunctionalId loop_xc_id,
                                           int ace_nocc,
                                           double ace_alpha_x,
                                           double exx_const_raw,
                                           int max_iters,
                                           double e_tol,
                                           double d_tol) -> bool {
    GpuHostKerkerDiis diis(ngrid, basis.g2(), vol, opts.mixing_beta, opts.kerker_q0, opts.diis_max_vec);
    double prev_etot = 1e100;
    bool loop_conv = false;
    DeviceBuffer<double> d_rho_out(static_cast<std::size_t>(ngrid));
    std::vector<Complex> h_rho_in_g(static_cast<std::size_t>(ngrid));
    std::vector<Complex> h_rho_out_g(static_cast<std::size_t>(ngrid));

    for (int iter = 0; iter < max_iters; ++iter) {
      ++res.scf_iterations;
      double eh_in = 0.0, exc_in = 0.0;
      eval_hartree_xc_veff_gpu(d_rho_r, loop_xc_id, eh_in, exc_in);
      d_rho_g.download(reinterpret_cast<cuDoubleComplex*>(h_rho_in_g.data()), h_rho_in_g.size());

      const double eig_tol_iter =
          (iter == 0 && d_psi_k[0].size() == 0)
              ? std::max(opts.eig_tol, 5e-4)
              : std::clamp(0.02 * std::max(res.final_density_rms, 1e-8), opts.eig_tol, 1e-5);

      for (int ik = 0; ik < nkpts; ++ik) {
        auto apply_h = [&](const cuDoubleComplex* d_x, cuDoubleComplex* d_hx, int nb) {
          apply_pw_hamiltonian_gpu(ctx, kp_gpu[ik], d_veff_r.data(), ace_nocc, ace_alpha_x,
                                   d_x, d_hx, nb, d_fft_work, d_proj_work);
        };
        auto eig_res = run_eigensolver_gpu(
            ctx, basis.kpoint(ik), kp_gpu[ik].d_kin.data(), nband_out, n_solve,
            opts.eigensolver, eig_tol_iter, opts.max_eig_iter, apply_h, d_psi_k[ik], false);
        res.total_eig_iterations += eig_res.iterations;
        res.total_hpsi_calls += eig_res.hpsi_calls;
        evals_k[ik] = std::move(eig_res.eigenvalues);
      }

      compute_density_gpu(d_rho_out);

      double exx_curr = exx_const_raw;
      if (ace_nocc > 0 && std::abs(ace_alpha_x) > 0.0) {
        exx_curr = 0.0;
        const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
        DeviceBuffer<cuDoubleComplex> d_ov(static_cast<std::size_t>(ace_nocc) * ace_nocc);
        std::vector<Complex> h_ov(d_ov.size());
        for (int ik = 0; ik < nkpts; ++ik) {
          const int npw = kp_gpu[ik].npw;
          ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, ace_nocc, ace_nocc, npw, one,
                    kp_gpu[ik].d_ace_xi.data(), npw, d_psi_k[ik].data(), npw,
                    zero, d_ov.data(), ace_nocc);
          d_ov.download(reinterpret_cast<cuDoubleComplex*>(h_ov.data()), h_ov.size());
          double fn2 = 0.0;
          for (const auto& z : h_ov) fn2 += std::norm(z);
          exx_curr -= kp_gpu[ik].weight * fn2;
        }
      }

      compute_variational_components_gpu(d_rho_out, loop_xc_id, exx_curr);
      // Note: compute_variational_components_gpu just placed rho_out_g into d_rho_g!
      d_rho_g.download(reinterpret_cast<cuDoubleComplex*>(h_rho_out_g.data()), h_rho_out_g.size());

      const double de = std::abs(res.total_energy - prev_etot);
      prev_etot = res.total_energy;

      const double d_rms = diis.step(ctx, static_cast<double>(n_elec),
                                     h_rho_in_g.data(), h_rho_out_g.data(), d_buf_c, d_rho_r);
      res.final_density_rms = d_rms;
      if (iter >= 1 && de < e_tol && d_rms < d_tol) {
        check_cuda(cudaMemcpy(d_rho_r.data(), d_rho_out.data(),
                              static_cast<std::size_t>(ngrid) * sizeof(double),
                              cudaMemcpyDeviceToDevice),
                   "converged rho_r");
        loop_conv = true;
        break;
      }
    }
    return loop_conv;
  };

  std::vector<DeviceBuffer<double>> d_coul_table;
  if (is_hybrid) {
    d_coul_table.resize(static_cast<std::size_t>(nkpts * nkpts));
    for (int ik = 0; ik < nkpts; ++ik) {
      const auto& kp_t = basis.kpoint(ik);
      for (int iq = 0; iq < nkpts; ++iq) {
        const auto& kp_q = basis.kpoint(iq);
        const std::array<double, 3> q_shift{
            kp_t.k_cart[0] - kp_q.k_cart[0],
            kp_t.k_cart[1] - kp_q.k_cart[1],
            kp_t.k_cart[2] - kp_q.k_cart[2],
        };
        const auto h_coul = basis.compute_exx_coulomb_kernel(q_shift, opts.exxdiv, opts.kmesh);
        d_coul_table[static_cast<std::size_t>(ik * nkpts + iq)].upload(h_coul.data(), h_coul.size());
      }
    }
  }

  if (!is_hybrid) {
    res.converged = run_semilocal_or_ace_loop_gpu(
        xc_id, 0, 0.0, 0.0, opts.max_scf_iter, opts.conv_tol, opts.density_tol);
  } else if (opts.use_ace) {
    const auto pre_xc = (xc_id == xc_gen::XcFunctionalId::k_hf)
                            ? xc_gen::XcFunctionalId::k_svwn
                            : xc_gen::XcFunctionalId::k_pbe;
    run_semilocal_or_ace_loop_gpu(pre_xc, 0, 0.0, 0.0, 6, 1e-4, 1e-3);

    // Unified Kerker-DIIS + Lin (2016) GPU ACE loop:
    // Builds low-rank ACE projector Xi_k via symmetric batched Poisson cuFFTs + Cholesky/TRSM,
    // and applies -Xi_k (Xi_k^H Psi) inside all Davidson/LOBPCG iterations via pure cublasZgemm!
    GpuHostKerkerDiis diis(ngrid, basis.g2(), vol, opts.mixing_beta, opts.kerker_q0, opts.diis_max_vec);
    double prev_etot = 1e100;
    const double inv_sqrt_vol = 1.0 / std::sqrt(vol);
    DeviceBuffer<double> d_rho_out(static_cast<std::size_t>(ngrid));
    std::vector<Complex> h_rho_in_g(static_cast<std::size_t>(ngrid));
    std::vector<Complex> h_rho_out_g(static_cast<std::size_t>(ngrid));
    std::vector<DeviceBuffer<cuDoubleComplex>> d_u_occ_r(static_cast<std::size_t>(nkpts));
    std::vector<DeviceBuffer<cuDoubleComplex>> d_psi_occ_prev(static_cast<std::size_t>(nkpts));
    DeviceBuffer<cuDoubleComplex> d_ov(static_cast<std::size_t>(n_occ) * n_occ);
    std::vector<Complex> h_ov(d_ov.size());

    for (int iter = 0; iter < opts.max_scf_iter; ++iter) {
      ++res.scf_iterations;

      double eh_in = 0.0, exc_in = 0.0;
      eval_hartree_xc_veff_gpu(d_rho_r, xc_id, eh_in, exc_in);
      d_rho_g.download(reinterpret_cast<cuDoubleComplex*>(h_rho_in_g.data()), h_rho_in_g.size());

      const bool do_rebuild = true;
      if (do_rebuild) {
        ++res.ace_iterations;
        for (int iq = 0; iq < nkpts; ++iq) {
          const int npw = kp_gpu[iq].npw;
          d_u_occ_r[iq].resize(static_cast<std::size_t>(n_occ) * ngrid);
          d_u_occ_r[iq].zero();
          pw_scatter_sphere_to_box_kernel<<<(npw * n_occ + threads - 1) / threads, threads>>>(
              npw, n_occ, ngrid, kp_gpu[iq].d_fft_idx.data(),
              d_psi_k[iq].data(), d_u_occ_r[iq].data());
          ctx.exec_fft3d(n_occ, d_u_occ_r[iq].data(), d_u_occ_r[iq].data(), +1);
          pw_scale_complex_kernel<<<static_cast<unsigned int>((d_u_occ_r[iq].size() + threads - 1) / threads), threads>>>(
              d_u_occ_r[iq].size(), inv_sqrt_vol, d_u_occ_r[iq].data());
        }

        for (int ik = 0; ik < nkpts; ++ik) {
          const int npw = kp_gpu[ik].npw;
          DeviceBuffer<cuDoubleComplex> d_W(static_cast<std::size_t>(npw) * n_occ);
          apply_exact_exchange_poisson_gpu(
              ctx, basis, ik, n_occ, kp_gpu, d_u_occ_r, d_coul_table, d_psi_k[ik].data(), n_occ,
              true, d_W.data(), res.total_poisson_ffts);

          DeviceBuffer<cuDoubleComplex> d_B(static_cast<std::size_t>(n_occ) * n_occ);
          ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, n_occ, n_occ, npw,
                    make_cuDoubleComplex(-1.0, 0.0), d_psi_k[ik].data(), npw,
                    d_W.data(), npw, make_cuDoubleComplex(0.0, 0.0), d_B.data(), n_occ);

          kp_gpu[ik].d_ace_xi.resize(static_cast<std::size_t>(npw) * n_occ);
          cholesky_and_trsm_ace_gpu(ctx, npw, n_occ, d_B.data(), d_W.data(), kp_gpu[ik].d_ace_xi.data());

          d_psi_occ_prev[ik].resize(static_cast<std::size_t>(npw) * n_occ);
          check_cuda(cudaMemcpy(d_psi_occ_prev[ik].data(), d_psi_k[ik].data(),
                                static_cast<std::size_t>(npw) * n_occ * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice),
                     "save psi_occ_prev");
        }
      }

      const double eig_tol_iter =
          std::clamp(0.02 * std::max(res.final_density_rms, 1e-8), opts.eig_tol, 1e-5);
      for (int ik = 0; ik < nkpts; ++ik) {
        auto apply_h = [&](const cuDoubleComplex* d_x, cuDoubleComplex* d_hx, int nb) {
          apply_pw_hamiltonian_gpu(ctx, kp_gpu[ik], d_veff_r.data(), n_occ, alpha_x,
                                   d_x, d_hx, nb, d_fft_work, d_proj_work);
        };
        auto eig_res = run_eigensolver_gpu(
            ctx, basis.kpoint(ik), kp_gpu[ik].d_kin.data(), nband_out, n_solve,
            opts.eigensolver, eig_tol_iter, opts.max_eig_iter, apply_h, d_psi_k[ik], false);
        res.total_eig_iterations += eig_res.iterations;
        res.total_hpsi_calls += eig_res.hpsi_calls;
        evals_k[ik] = std::move(eig_res.eigenvalues);
      }

      double exx_proj = 0.0;
      double subspace_err2 = 0.0;
      const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
      const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
      for (int ik = 0; ik < nkpts; ++ik) {
        const int npw = kp_gpu[ik].npw;
        ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, n_occ, n_occ, npw, one,
                  kp_gpu[ik].d_ace_xi.data(), npw, d_psi_k[ik].data(), npw,
                  zero, d_ov.data(), n_occ);
        d_ov.download(reinterpret_cast<cuDoubleComplex*>(h_ov.data()), h_ov.size());
        double fn2 = 0.0;
        for (const auto& z : h_ov) fn2 += std::norm(z);
        exx_proj -= kp_gpu[ik].weight * fn2;

        ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, n_occ, n_occ, npw, one,
                  d_psi_occ_prev[ik].data(), npw, d_psi_k[ik].data(), npw,
                  zero, d_ov.data(), n_occ);
        d_ov.download(reinterpret_cast<cuDoubleComplex*>(h_ov.data()), h_ov.size());
        double proj_norm2 = 0.0;
        for (const auto& z : h_ov) proj_norm2 += std::norm(z);
        subspace_err2 += kp_gpu[ik].weight *
                         std::max(0.0, static_cast<double>(n_occ) - proj_norm2);
      }

      compute_density_gpu(d_rho_out);
      compute_variational_components_gpu(d_rho_out, xc_id, exx_proj);
      d_rho_g.download(reinterpret_cast<cuDoubleComplex*>(h_rho_out_g.data()), h_rho_out_g.size());

      const double de = std::abs(res.total_energy - prev_etot);
      prev_etot = res.total_energy;

      const double d_rms = diis.step(ctx, static_cast<double>(n_elec),
                                     h_rho_in_g.data(), h_rho_out_g.data(), d_buf_c, d_rho_r);
      res.final_density_rms = d_rms;
      if (iter >= 2 && do_rebuild && de < opts.conv_tol && subspace_err2 < opts.conv_tol &&
          d_rms < opts.density_tol) {
        check_cuda(cudaMemcpy(d_rho_r.data(), d_rho_out.data(),
                              static_cast<std::size_t>(ngrid) * sizeof(double),
                              cudaMemcpyDeviceToDevice),
                   "converged ace rho_r");
        res.converged = true;
        break;
      }
    }
  } else {
    // Uncompressed GPU Exact Exchange baseline (evaluates batched 3D Poisson cuFFTs inside EVERY Davidson/LOBPCG H*psi!)
    const auto pre_xc = (xc_id == xc_gen::XcFunctionalId::k_hf)
                            ? xc_gen::XcFunctionalId::k_svwn
                            : xc_gen::XcFunctionalId::k_pbe;
    run_semilocal_or_ace_loop_gpu(pre_xc, 0, 0.0, 0.0, 6, 1e-4, 1e-3);

    GpuHostKerkerDiis diis(ngrid, basis.g2(), vol, opts.mixing_beta, opts.kerker_q0, opts.diis_max_vec);
    double prev_etot = 1e100;
    const double inv_sqrt_vol = 1.0 / std::sqrt(vol);
    DeviceBuffer<double> d_rho_out(static_cast<std::size_t>(ngrid));
    std::vector<Complex> h_rho_in_g(static_cast<std::size_t>(ngrid));
    std::vector<Complex> h_rho_out_g(static_cast<std::size_t>(ngrid));
    std::vector<DeviceBuffer<cuDoubleComplex>> d_u_occ_r(static_cast<std::size_t>(nkpts));

    for (int iter = 0; iter < opts.max_scf_iter; ++iter) {
      ++res.scf_iterations;
      double eh_in = 0.0, exc_in = 0.0;
      eval_hartree_xc_veff_gpu(d_rho_r, xc_id, eh_in, exc_in);
      d_rho_g.download(reinterpret_cast<cuDoubleComplex*>(h_rho_in_g.data()), h_rho_in_g.size());

      for (int iq = 0; iq < nkpts; ++iq) {
        const int npw = kp_gpu[iq].npw;
        d_u_occ_r[iq].resize(static_cast<std::size_t>(n_occ) * ngrid);
        d_u_occ_r[iq].zero();
        pw_scatter_sphere_to_box_kernel<<<(npw * n_occ + threads - 1) / threads, threads>>>(
            npw, n_occ, ngrid, kp_gpu[iq].d_fft_idx.data(),
            d_psi_k[iq].data(), d_u_occ_r[iq].data());
        ctx.exec_fft3d(n_occ, d_u_occ_r[iq].data(), d_u_occ_r[iq].data(), +1);
        pw_scale_complex_kernel<<<static_cast<unsigned int>((d_u_occ_r[iq].size() + threads - 1) / threads), threads>>>(
            d_u_occ_r[iq].size(), inv_sqrt_vol, d_u_occ_r[iq].data());
      }

      const double eig_tol_iter =
          std::clamp(0.02 * std::max(res.final_density_rms, 1e-8), opts.eig_tol, 1e-5);
      for (int ik = 0; ik < nkpts; ++ik) {
        const int npw = kp_gpu[ik].npw;
        DeviceBuffer<cuDoubleComplex> d_vx_work;
        auto apply_h = [&](const cuDoubleComplex* d_x, cuDoubleComplex* d_hx, int nb) {
          apply_pw_hamiltonian_gpu(ctx, kp_gpu[ik], d_veff_r.data(), 0, 0.0,
                                   d_x, d_hx, nb, d_fft_work, d_proj_work);
          d_vx_work.resize(static_cast<std::size_t>(npw) * nb);
          apply_exact_exchange_poisson_gpu(
              ctx, basis, ik, n_occ, kp_gpu, d_u_occ_r, d_coul_table, d_x, nb,
              false, d_vx_work.data(), res.total_poisson_ffts);
          const cuDoubleComplex ax = make_cuDoubleComplex(alpha_x, 0.0);
          check_cublas(cublasZaxpy(ctx.cublas(), npw * nb, &ax, d_vx_work.data(), 1, d_hx, 1),
                       "cublasZaxpy uncompressed exx");
        };
        auto eig_res = run_eigensolver_gpu(
            ctx, basis.kpoint(ik), kp_gpu[ik].d_kin.data(), nband_out, n_solve,
            opts.eigensolver, eig_tol_iter, opts.max_eig_iter, apply_h, d_psi_k[ik], false);
        res.total_eig_iterations += eig_res.iterations;
        res.total_hpsi_calls += eig_res.hpsi_calls;
        evals_k[ik] = std::move(eig_res.eigenvalues);
      }

      for (int iq = 0; iq < nkpts; ++iq) {
        const int npw = kp_gpu[iq].npw;
        d_u_occ_r[iq].zero();
        pw_scatter_sphere_to_box_kernel<<<(npw * n_occ + threads - 1) / threads, threads>>>(
            npw, n_occ, ngrid, kp_gpu[iq].d_fft_idx.data(),
            d_psi_k[iq].data(), d_u_occ_r[iq].data());
        ctx.exec_fft3d(n_occ, d_u_occ_r[iq].data(), d_u_occ_r[iq].data(), +1);
        pw_scale_complex_kernel<<<static_cast<unsigned int>((d_u_occ_r[iq].size() + threads - 1) / threads), threads>>>(
            d_u_occ_r[iq].size(), inv_sqrt_vol, d_u_occ_r[iq].data());
      }
      double exx_raw = 0.0;
      for (int ik = 0; ik < nkpts; ++ik) {
        const int npw = kp_gpu[ik].npw;
        DeviceBuffer<cuDoubleComplex> d_W(static_cast<std::size_t>(npw) * n_occ);
        apply_exact_exchange_poisson_gpu(
            ctx, basis, ik, n_occ, kp_gpu, d_u_occ_r, d_coul_table, d_psi_k[ik].data(), n_occ,
            (nkpts == 1), d_W.data(), res.total_poisson_ffts);
        DeviceBuffer<cuDoubleComplex> d_B(static_cast<std::size_t>(n_occ) * n_occ);
        ctx.zgemm(CUBLAS_OP_C, CUBLAS_OP_N, n_occ, n_occ, npw,
                  make_cuDoubleComplex(1.0, 0.0), d_psi_k[ik].data(), npw,
                  d_W.data(), npw, make_cuDoubleComplex(0.0, 0.0), d_B.data(), n_occ);
        std::vector<Complex> h_B(d_B.size());
        d_B.download(reinterpret_cast<cuDoubleComplex*>(h_B.data()), h_B.size());
        double tr = 0.0;
        for (int j = 0; j < n_occ; ++j) tr += h_B[j + static_cast<std::size_t>(j) * n_occ].real();
        exx_raw += kp_gpu[ik].weight * tr;
      }

      compute_density_gpu(d_rho_out);
      compute_variational_components_gpu(d_rho_out, xc_id, exx_raw);
      d_rho_g.download(reinterpret_cast<cuDoubleComplex*>(h_rho_out_g.data()), h_rho_out_g.size());

      const double de = std::abs(res.total_energy - prev_etot);
      prev_etot = res.total_energy;
      const double d_rms = diis.step(ctx, static_cast<double>(n_elec),
                                     h_rho_in_g.data(), h_rho_out_g.data(), d_buf_c, d_rho_r);
      res.final_density_rms = d_rms;
      if (iter >= 1 && de < opts.conv_tol && d_rms < opts.density_tol) {
        check_cuda(cudaMemcpy(d_rho_r.data(), d_rho_out.data(),
                              static_cast<std::size_t>(ngrid) * sizeof(double),
                              cudaMemcpyDeviceToDevice),
                   "converged uncompressed rho_r");
        res.converged = true;
        break;
      }
    }
  }

  double homo = -1e100;
  double lumo = +1e100;
  for (int ik = 0; ik < nkpts; ++ik) {
    if (n_occ > 0 && static_cast<int>(evals_k[ik].size()) >= n_occ) {
      homo = std::max(homo, evals_k[ik][n_occ - 1]);
    }
    if (static_cast<int>(evals_k[ik].size()) > n_occ) {
      lumo = std::min(lumo, evals_k[ik][n_occ]);
    }
  }
  res.homo_energy = (homo > -1e90) ? homo : 0.0;
  res.lumo_energy = (lumo < +1e90) ? lumo : 0.0;
  res.eigenvalues = std::move(evals_k);
  res.density_r.resize(static_cast<std::size_t>(ngrid));
  d_rho_r.download(res.density_r.data(), res.density_r.size());

  check_cuda(cudaDeviceSynchronize(), "final sync");
  const auto t_end = std::chrono::steady_clock::now();
  res.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  return res;
}

}  // namespace gpu4dft::pw::detail
