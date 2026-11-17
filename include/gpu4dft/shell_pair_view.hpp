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
//   prefactor = c_a c_b Y_00^2 exp(-a b / p |A - B|^2) (pi / p)^{3/2}.
struct ShellPairView {
  int npair = 0;
  const int* prim_offset = nullptr;     // [npair + 1]
  const double* exponent = nullptr;     // [nprim_pair]
  const double* Px = nullptr;           // [nprim_pair]
  const double* Py = nullptr;           // [nprim_pair]
  const double* Pz = nullptr;           // [nprim_pair]
  const double* prefactor = nullptr;    // [nprim_pair]

  // Extended per-pair [npair] and per-primitive [nprim_pair] metadata for
  // arbitrary angular momentum, Schwarz screening, direct J/K, and gradients.
  const double* Ax = nullptr;           // [npair]
  const double* Ay = nullptr;           // [npair]
  const double* Az = nullptr;           // [npair]
  const double* Bx = nullptr;           // [npair]
  const double* By = nullptr;           // [npair]
  const double* Bz = nullptr;           // [npair]
  const double* alpha_a = nullptr;      // [nprim_pair]
  const double* alpha_b = nullptr;      // [nprim_pair]
  const double* schwarz_bound = nullptr;// [npair] sqrt(max |(uv|uv)|), sorted descending
  const int* ish = nullptr;             // [npair]
  const int* jsh = nullptr;             // [npair]
  const int* ao_i = nullptr;            // [npair]
  const int* ao_j = nullptr;            // [npair]
};

}  // namespace gpu4dft
