#pragma once

#include <span>

#include "gpu4dft/shell_pair_view.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

// Launcher of a generated ERI kernel; see gpu4dft/generated/eri_launchers.hpp.
using EriLaunchFn = void (*)(const ShellPairView& bra, const ShellPairView& ket, double* out,
                             void* stream);

struct EriKernel {
  int la, lb, lc, ld;
  Backend backend;
  Precision precision;
  EriLaunchFn launch;
  const char* name;
};

// All generated ERI kernels compiled into this build (defined in the generated
// src/eri_registry.cpp).
std::span<const EriKernel> eri_kernels();

}  // namespace gpu4dft
