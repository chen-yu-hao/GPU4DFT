#pragma once

namespace gpu4dft {

// Structure-of-arrays view of a shell-pair list, trivially copyable so it can
// be passed by value to GPU kernels. The pointers refer to host or device
// memory depending on the backend that consumes the view.
//
// Primitive pairs of shell pair i occupy [prim_offset[i], prim_offset[i + 1]).
// For primitive exponents a (on A) and b (on B):
//   exponent  = p = a + b
//   P{x,y,z}  = (a A + b B) / p
//   prefactor = c_a c_b exp(-a b / p |A - B|^2) (pi / p)^{3/2}, with the
//               contraction coefficients c and angular normalisation folded in.
struct ShellPairView {
  int npair = 0;
  const int* prim_offset = nullptr;  // [npair + 1]
  const double* exponent = nullptr;  // [nprim_pair]
  const double* Px = nullptr;
  const double* Py = nullptr;
  const double* Pz = nullptr;
  const double* prefactor = nullptr;
};

}  // namespace gpu4dft
