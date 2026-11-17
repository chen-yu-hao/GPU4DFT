#pragma once

#include <vector>

#include "gpu4dft/kernel_registry.hpp"
#include "gpu4dft/shell_pairs.hpp"

// CUDA runtime glue. Only compiled when GPU4DFT_HAVE_CUDA is defined; keeps
// CUDA headers out of the host-only translation units.
namespace gpu4dft::detail {

// Number of visible CUDA devices; 0 if there is no driver or no device.
int cuda_device_count();

// Uploads both pair lists, runs `launch` on the default stream and copies the
// result into `out`, which must already have size bra.npair() * ket.npair().
void run_eri_kernel_cuda(EriLaunchFn launch, const ShellPairData& bra, const ShellPairData& ket,
                         std::vector<double>& out);

}  // namespace gpu4dft::detail
