#pragma once

#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/shell_pair_view.hpp"

namespace gpu4dft {

// Host-side owner of a shell-pair list for one angular-momentum class (la, lb).
struct ShellPairData {
  int la = 0;
  int lb = 0;
  std::vector<int> ish;  // [npair] first shell of each pair
  std::vector<int> jsh;  // [npair] second shell of each pair
  std::vector<int> prim_offset{0};
  std::vector<double> exponent;
  std::vector<double> Px;
  std::vector<double> Py;
  std::vector<double> Pz;
  std::vector<double> prefactor;

  int npair() const { return static_cast<int>(ish.size()); }
  // View over host memory.
  ShellPairView view() const;
};

// Builds the pairs (i, j) of shells with l_i = la and l_j = lb; when la == lb
// only i >= j is kept. Primitive pairs with |prefactor| < prim_cutoff are
// dropped. Only (s, s) pairs are implemented so far; other classes throw
// gpu4dft::NotImplemented.
ShellPairData build_shell_pairs(const Basis& basis, int la, int lb, double prim_cutoff = 0.0);

}  // namespace gpu4dft
