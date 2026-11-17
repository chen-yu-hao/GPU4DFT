#pragma once

#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/shell_pairs.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

// True if `backend` was compiled in and has a usable device.
bool backend_available(Backend backend);

// Contracted integrals (bra pair | ket pair) for every bra and ket shell pair,
// row-major [bra.npair() x ket.npair()]. Throws gpu4dft::NotImplemented if no
// kernel exists for the angular-momentum class.
std::vector<double> eri_pair_block(const ShellPairData& bra, const ShellPairData& ket,
                                   Backend backend, Precision precision);

// Full ERI tensor (ij|kl) in chemists' notation, row-major [nao]^4 in the AO
// order of `basis`. O(nao^4) memory: meant for validation, not production.
std::vector<double> eri_tensor(const Basis& basis, Backend backend, Precision precision);

}  // namespace gpu4dft
