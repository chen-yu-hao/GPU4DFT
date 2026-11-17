#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/one_electron.hpp"

namespace gpu4dft::detail {
namespace {

void check_cuda_1e(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DevBuf1e {
 public:
  DevBuf1e() = default;
  explicit DevBuf1e(std::size_t n) : n_(n) {
    if (n_ > 0) check_cuda_1e(cudaMalloc(&ptr_, n_ * sizeof(T)), "cudaMalloc (1e)");
  }
  DevBuf1e(const T* host, std::size_t n) : DevBuf1e(n) {
    if (n_ > 0) {
      check_cuda_1e(cudaMemcpy(ptr_, host, n_ * sizeof(T), cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D (1e)");
    }
  }
  ~DevBuf1e() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  DevBuf1e(const DevBuf1e&) = delete;
  DevBuf1e& operator=(const DevBuf1e&) = delete;

  T* data() const { return ptr_; }
  std::size_t size() const { return n_; }

 private:
  T* ptr_ = nullptr;
  std::size_t n_ = 0;
};

struct UploadedBasis1e {
  DevBuf1e<int> shell_l;
  DevBuf1e<int> shell_ao;
  DevBuf1e<int> shell_atom;
  DevBuf1e<int> prim_offset;
  DevBuf1e<double> shell_center;
  DevBuf1e<double> prim_exp;
  DevBuf1e<double> prim_coeff;
  DevBuf1e<int> atom_Z;
  DevBuf1e<double> atom_center;
  BasisDeviceView view;

  explicit UploadedBasis1e(const BasisDeviceView& h)
      : shell_l(h.shell_l, h.nshell),
        shell_ao(h.shell_ao, h.nshell),
        shell_atom(h.shell_atom, h.nshell),
        prim_offset(h.prim_offset, h.nshell + 1),
        shell_center(h.shell_center, static_cast<std::size_t>(h.nshell) * 3),
        prim_exp(h.prim_exp, h.prim_offset[h.nshell]),
        prim_coeff(h.prim_coeff, h.prim_offset[h.nshell]),
        atom_Z(h.atom_Z, h.natom),
        atom_center(h.atom_center, static_cast<std::size_t>(h.natom) * 3) {
    view = BasisDeviceView{
        h.nshell,
        h.nao,
        h.natom,
        shell_l.data(),
        shell_ao.data(),
        shell_atom.data(),
        prim_offset.data(),
        shell_center.data(),
        prim_exp.data(),
        prim_coeff.data(),
        atom_Z.data(),
        atom_center.data(),
    };
  }
};

__global__ void one_electron_matrix_kernel(BasisDeviceView b, const int* __restrict__ pair_i,
                                           const int* __restrict__ pair_j, int npairs, int op,
                                           double* __restrict__ mat) {
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < npairs; idx += stride) {
    eval_one_elec_pair(b, pair_i[idx], pair_j[idx], op, mat);
  }
}

__global__ void one_electron_grad_kernel(BasisDeviceView b, const double* __restrict__ dm,
                                         const double* __restrict__ wdm,
                                         double* __restrict__ grad) {
  const int total_pairs = b.nshell * b.nshell;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_pairs; idx += stride) {
    const int ish = idx / b.nshell;
    const int jsh = idx - ish * b.nshell;
    double bra_f[3] = {0.0, 0.0, 0.0};
    auto add_hf = [&](int c, double fx, double fy, double fz) {
      atomicAdd(&grad[3 * c + 0], fx);
      atomicAdd(&grad[3 * c + 1], fy);
      atomicAdd(&grad[3 * c + 2], fz);
    };
    eval_one_elec_grad_ordered_pair(b, ish, jsh, dm, wdm, bra_f, add_hf);
    const int at_i = b.shell_atom[ish];
    if (at_i >= 0) {
      atomicAdd(&grad[3 * at_i + 0], bra_f[0]);
      atomicAdd(&grad[3 * at_i + 1], bra_f[1]);
      atomicAdd(&grad[3 * at_i + 2], bra_f[2]);
    }
  }
}

}  // namespace

void run_one_electron_matrix_cuda(const BasisDeviceView& host_view, int op,
                                  std::vector<double>& out) {
  UploadedBasis1e dev(host_view);
  const int nsh = host_view.nshell;
  std::vector<int> pi, pj;
  pi.reserve(static_cast<std::size_t>(nsh) * (nsh + 1) / 2);
  pj.reserve(pi.capacity());
  for (int i = 0; i < nsh; ++i) {
    for (int j = 0; j <= i; ++j) {
      pi.push_back(i);
      pj.push_back(j);
    }
  }
  const int npairs = static_cast<int>(pi.size());
  if (npairs == 0) return;

  DevBuf1e<int> d_pi(pi.data(), pi.size());
  DevBuf1e<int> d_pj(pj.data(), pj.size());
  DevBuf1e<double> d_mat(out.size());
  check_cuda_1e(cudaMemset(d_mat.data(), 0, out.size() * sizeof(double)), "cudaMemset (1e)");

  constexpr int kBlock = 64;
  const int blocks = std::min((npairs + kBlock - 1) / kBlock, 4096);
  one_electron_matrix_kernel<<<blocks, kBlock>>>(dev.view, d_pi.data(), d_pj.data(), npairs, op,
                                                 d_mat.data());
  check_cuda_1e(cudaGetLastError(), "one_electron_matrix_kernel launch");
  check_cuda_1e(cudaDeviceSynchronize(), "one_electron_matrix_kernel sync");
  check_cuda_1e(cudaMemcpy(out.data(), d_mat.data(), out.size() * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H (1e)");
}

void run_one_electron_grad_cuda(const BasisDeviceView& host_view, const double* dm,
                                const double* wdm, std::vector<double>& grad_out) {
  UploadedBasis1e dev(host_view);
  const std::size_t n2 = static_cast<std::size_t>(host_view.nao) * host_view.nao;
  DevBuf1e<double> d_dm(dm, n2);
  DevBuf1e<double> d_wdm(wdm, n2);
  DevBuf1e<double> d_grad(grad_out.size());
  check_cuda_1e(cudaMemset(d_grad.data(), 0, grad_out.size() * sizeof(double)),
                "cudaMemset (1e grad)");

  const int total_pairs = host_view.nshell * host_view.nshell;
  constexpr int kBlock = 64;
  const int blocks = std::min((total_pairs + kBlock - 1) / kBlock, 4096);
  one_electron_grad_kernel<<<blocks, kBlock>>>(dev.view, d_dm.data(), d_wdm.data(), d_grad.data());
  check_cuda_1e(cudaGetLastError(), "one_electron_grad_kernel launch");
  check_cuda_1e(cudaDeviceSynchronize(), "one_electron_grad_kernel sync");
  check_cuda_1e(cudaMemcpy(grad_out.data(), d_grad.data(), grad_out.size() * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H (1e grad)");
}

}  // namespace gpu4dft::detail
