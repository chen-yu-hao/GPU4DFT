#pragma once

#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/shell_pairs.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

// True if `backend` was compiled in and has a usable device.
bool backend_available(Backend backend);

// Contracted spherical integrals (bra pair | ket pair) for every bra and ket
// shell pair, row-major [bra.npair() * ket.npair() * n_sph(la)*n_sph(lb)*n_sph(lc)*n_sph(ld)].
// Throws gpu4dft::NotImplemented if angular momentum exceeds supported max_l (4).
std::vector<double> eri_pair_block(const ShellPairData& bra, const ShellPairData& ket,
                                   Backend backend, Precision precision, double omega = 0.0);

// Full ERI tensor (ij|kl) in chemists' notation, row-major [nao]^4 in the
// spherical AO order of `basis`. O(nao^4) memory: meant for validation.
std::vector<double> eri_tensor(const Basis& basis, Backend backend, Precision precision,
                               double omega = 0.0);

}  // namespace gpu4dft
