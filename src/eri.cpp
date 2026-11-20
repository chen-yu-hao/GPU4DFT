#include "gpu4dft/eri.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/generated/eri_launchers.hpp"
#include "gpu4dft/harmonics.hpp"
#include "gpu4dft/kernel_registry.hpp"

#if defined(GPU4DFT_HAVE_CUDA)
#include "cuda/eri_cuda.hpp"
#endif

namespace gpu4dft {

const char* to_string(Backend backend) {
  switch (backend) {
    case Backend::Host: return "host";
    case Backend::Cuda: return "cuda";
    case Backend::Hip: return "hip";
  }
  return "unknown";
}

const char* to_string(Precision precision) {
  switch (precision) {
    case Precision::FP64: return "fp64";
    case Precision::FP32: return "fp32";
    case Precision::Mixed: return "mixed";
  }
  return "unknown";
}

namespace {

constexpr int kMaxSupportedL = 4;

std::string class_name(int la, int lb, int lc, int ld) {
  constexpr const char* kLetters = "spdfghik";
  const int ls[4] = {la, lb, lc, ld};
  std::string name = "(";
  for (int idx = 0; idx < 4; ++idx) {
    if (idx == 2) name += '|';
    name += (ls[idx] >= 0 && ls[idx] < 8) ? kLetters[ls[idx]] : '?';
  }
  return name + ")";
}

const EriKernel* find_specialized_eri_kernel(int la, int lb, int lc, int ld, Backend backend,
                                             Precision precision) {
  const auto kernels = eri_kernels();
  for (const EriKernel& k : kernels) {
    if (k.la == la && k.lb == lb && k.lc == lc && k.ld == ld && k.backend == backend &&
        k.precision == precision) {
      return &k;
    }
  }
  return nullptr;
}

}  // namespace

bool backend_available(Backend backend) {
  switch (backend) {
    case Backend::Host: return true;
    case Backend::Cuda:
#if defined(GPU4DFT_HAVE_CUDA)
      return detail::cuda_device_count() > 0;
#else
      return false;
#endif
    case Backend::Hip:
#if defined(GPU4DFT_HAVE_HIP)
      return true;
#else
      return false;
#endif
  }
  return false;
}

std::vector<double> eri_pair_block(const ShellPairData& bra, const ShellPairData& ket,
                                   Backend backend, Precision precision, double omega) {
  const int la = bra.la, lb = bra.lb, lc = ket.la, ld = ket.lb;
  if (la < 0 || lb < 0 || lc < 0 || ld < 0 || la > kMaxSupportedL || lb > kMaxSupportedL ||
      lc > kMaxSupportedL || ld > kMaxSupportedL) {
    throw NotImplemented("no ERI kernel for class " + class_name(la, lb, lc, ld) + " yet");
  }

  const Precision eval_prec = (precision == Precision::Mixed) ? Precision::FP64 : precision;
  const std::size_t nsph =
      static_cast<std::size_t>(n_sph(la)) * n_sph(lb) * n_sph(lc) * n_sph(ld);
  std::vector<double> out(static_cast<std::size_t>(bra.npair()) * ket.npair() * nsph, 0.0);
  if (out.empty()) return out;

  if (omega > 0.0) {
    switch (backend) {
      case Backend::Host:
        if (!generated::eri_general_omega_dispatch(la, lb, lc, ld, Backend::Host, eval_prec,
                                                   bra.view(), ket.view(), omega, out.data(),
                                                   nullptr)) {
          throw std::runtime_error("host general omega ERI kernel is unavailable");
        }
        break;
      case Backend::Cuda:
#if defined(GPU4DFT_HAVE_CUDA)
        {
          const auto fn = (eval_prec == Precision::FP64) ? &generated::eri_general_cuda_fp64_omega
                                                         : &generated::eri_general_cuda_fp32_omega;
          detail::run_eri_general_omega_cuda(fn, la, lb, lc, ld, bra, ket, omega, out);
        }
        break;
#else
        throw std::runtime_error("gpu4dft was built without CUDA");
#endif
      case Backend::Hip:
        throw std::runtime_error("gpu4dft was built without HIP");
    }
    return out;
  }

  const EriKernel* spec = find_specialized_eri_kernel(la, lb, lc, ld, backend, eval_prec);

  switch (backend) {
    case Backend::Host:
      if (spec != nullptr) {
        spec->launch(bra.view(), ket.view(), out.data(), nullptr);
      } else if (!generated::eri_general_dispatch(la, lb, lc, ld, Backend::Host, eval_prec,
                                                  bra.view(), ket.view(), out.data(), nullptr)) {
        throw std::runtime_error("host general ERI kernel is unavailable");
      }
      break;
    case Backend::Cuda:
#if defined(GPU4DFT_HAVE_CUDA)
      if (spec != nullptr) {
        detail::run_eri_kernel_cuda(spec->launch, bra, ket, out);
      } else {
        const auto fn = (eval_prec == Precision::FP64) ? &generated::eri_general_cuda_fp64
                                                       : &generated::eri_general_cuda_fp32;
        detail::run_eri_general_cuda(fn, la, lb, lc, ld, bra, ket, out);
      }
      break;
#else
      throw std::runtime_error("gpu4dft was built without CUDA");
#endif
    case Backend::Hip:
      throw std::runtime_error("gpu4dft was built without HIP");
  }
  return out;
}

std::vector<double> eri_tensor(const Basis& basis, Backend backend, Precision precision,
                               double omega) {
  if (basis.max_l() > kMaxSupportedL) {
    throw NotImplemented("eri_tensor supports up to g shells (l <= 4); basis has l = " +
                         std::to_string(basis.max_l()));
  }
  if (!backend_available(backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(backend) + "' is not available");
  }

  const int max_l = basis.max_l();
  std::vector<ShellPairData> pair_lists;
  for (int la = 0; la <= max_l; ++la) {
    for (int lb = 0; lb <= la; ++lb) {
      ShellPairData pd = build_shell_pairs(basis, la, lb, 0.0, omega);
      if (pd.npair() > 0) {
        pair_lists.push_back(std::move(pd));
      }
    }
  }

  const std::size_t n = static_cast<std::size_t>(basis.nao());
  std::vector<double> eri(n * n * n * n, 0.0);
  auto at = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) -> double& {
    return eri[((i * n + j) * n + k) * n + l];
  };

  for (std::size_t bp = 0; bp < pair_lists.size(); ++bp) {
    const ShellPairData& bra = pair_lists[bp];
    const int nsa = n_sph(bra.la);
    const int nsb = n_sph(bra.lb);

    for (std::size_t kp = 0; kp <= bp; ++kp) {
      const ShellPairData& ket = pair_lists[kp];
      const int nsc = n_sph(ket.la);
      const int nsd = n_sph(ket.lb);
      const std::size_t nsph = static_cast<std::size_t>(nsa) * nsb * nsc * nsd;
      const std::size_t nket = static_cast<std::size_t>(ket.npair());

      const std::vector<double> block = eri_pair_block(bra, ket, backend, precision, omega);

      for (std::size_t P = 0; P < static_cast<std::size_t>(bra.npair()); ++P) {
        const std::size_t i0 = static_cast<std::size_t>(bra.ao_i[P]);
        const std::size_t j0 = static_cast<std::size_t>(bra.ao_j[P]);
        for (std::size_t Q = 0; Q < nket; ++Q) {
          if (bp == kp && Q > P) continue;
          const std::size_t k0 = static_cast<std::size_t>(ket.ao_i[Q]);
          const std::size_t l0 = static_cast<std::size_t>(ket.ao_j[Q]);
          const double* qptr = block.data() + (P * nket + Q) * nsph;

          for (int ma = 0; ma < nsa; ++ma) {
            const std::size_t i = i0 + ma;
            for (int mb = 0; mb < nsb; ++mb) {
              const std::size_t j = j0 + mb;
              for (int mc = 0; mc < nsc; ++mc) {
                const std::size_t k = k0 + mc;
                for (int md = 0; md < nsd; ++md) {
                  const std::size_t l = l0 + md;
                  const double v = qptr[((ma * nsb + mb) * nsc + mc) * nsd + md];
                  at(i, j, k, l) = v;
                  at(j, i, k, l) = v;
                  at(i, j, l, k) = v;
                  at(j, i, l, k) = v;
                  at(k, l, i, j) = v;
                  at(l, k, i, j) = v;
                  at(k, l, j, i) = v;
                  at(l, k, j, i) = v;
                }
              }
            }
          }
        }
      }
    }
  }
  return eri;
}

}  // namespace gpu4dft
