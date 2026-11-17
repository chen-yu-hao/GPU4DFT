#include "gpu4dft/eri.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "gpu4dft/kernel_registry.hpp"

#if defined(GPU4DFT_HAVE_CUDA)
#include "cuda/eri_cuda.hpp"
#endif

namespace gpu4dft {

const char* to_string(Backend backend) {
  switch (backend) {
    case Backend::Host: return "host";
    case Backend::Cuda: return "cuda";
  }
  return "unknown";
}

const char* to_string(Precision precision) {
  switch (precision) {
    case Precision::FP64: return "fp64";
    case Precision::FP32: return "fp32";
  }
  return "unknown";
}

namespace {

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

const EriKernel& find_eri_kernel(int la, int lb, int lc, int ld, Backend backend,
                                 Precision precision) {
  const auto kernels = eri_kernels();
  bool class_generated = false;
  for (const EriKernel& k : kernels) {
    if (k.la != la || k.lb != lb || k.lc != lc || k.ld != ld) continue;
    class_generated = true;
    if (k.backend == backend && k.precision == precision) return k;
  }
  if (!class_generated) {
    throw NotImplemented("no ERI kernel for class " + class_name(la, lb, lc, ld) + " yet");
  }
  throw std::runtime_error(std::string("ERI kernels for backend '") + to_string(backend) +
                           "' are not compiled into this build");
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
  }
  return false;
}

std::vector<double> eri_pair_block(const ShellPairData& bra, const ShellPairData& ket,
                                   Backend backend, Precision precision) {
  const EriKernel& kernel = find_eri_kernel(bra.la, bra.lb, ket.la, ket.lb, backend, precision);
  std::vector<double> out(static_cast<std::size_t>(bra.npair()) * ket.npair());
  if (out.empty()) return out;

  switch (backend) {
    case Backend::Host:
      kernel.launch(bra.view(), ket.view(), out.data(), nullptr);
      break;
    case Backend::Cuda:
#if defined(GPU4DFT_HAVE_CUDA)
      detail::run_eri_kernel_cuda(kernel.launch, bra, ket, out);
      break;
#else
      // Unreachable: find_eri_kernel finds no CUDA kernels in a host-only build.
      throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }
  return out;
}

std::vector<double> eri_tensor(const Basis& basis, Backend backend, Precision precision) {
  if (basis.max_l() > 0) {
    throw NotImplemented("eri_tensor supports s shells only so far; basis has l = " +
                         std::to_string(basis.max_l()));
  }
  const ShellPairData pairs = build_shell_pairs(basis, 0, 0);
  const std::vector<double> block = eri_pair_block(pairs, pairs, backend, precision);

  const std::size_t n = static_cast<std::size_t>(basis.nao());
  const std::size_t npair = static_cast<std::size_t>(pairs.npair());
  std::vector<double> eri(n * n * n * n, 0.0);
  auto at = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) -> double& {
    return eri[((i * n + j) * n + k) * n + l];
  };
  // Every s shell holds one AO, so a shell pair maps to one AO pair. The block
  // covers bra and ket pairs with i >= j; expand the remaining permutations.
  for (std::size_t P = 0; P < npair; ++P) {
    const std::size_t i = basis.ao_offset(pairs.ish[P]);
    const std::size_t j = basis.ao_offset(pairs.jsh[P]);
    for (std::size_t Q = 0; Q < npair; ++Q) {
      const std::size_t k = basis.ao_offset(pairs.ish[Q]);
      const std::size_t l = basis.ao_offset(pairs.jsh[Q]);
      const double v = block[P * npair + Q];
      at(i, j, k, l) = v;
      at(j, i, k, l) = v;
      at(i, j, l, k) = v;
      at(j, i, l, k) = v;
    }
  }
  return eri;
}

}  // namespace gpu4dft
