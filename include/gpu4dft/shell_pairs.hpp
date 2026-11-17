#pragma once

#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/shell_pair_view.hpp"

namespace gpu4dft {

// Host-side owner of a SoA shell-pair list for one angular-momentum class
// (la, lb), sorted in descending order of `schwarz_bound`.
struct ShellPairData {
  int la = 0;
  int lb = 0;
  std::vector<int> ish;            // [npair] first shell of each pair
  std::vector<int> jsh;            // [npair] second shell of each pair
  std::vector<int> ao_i;           // [npair] spherical AO offset of ish
  std::vector<int> ao_j;           // [npair] spherical AO offset of jsh
  std::vector<double> Ax;          // [npair] center A of ish
  std::vector<double> Ay;
  std::vector<double> Az;
  std::vector<double> Bx;          // [npair] center B of jsh
  std::vector<double> By;
  std::vector<double> Bz;
  std::vector<double> schwarz_bound;  // [npair] sqrt(max_{u in i, v in j} |(uv|uv)|)

  std::vector<int> prim_offset{0}; // [npair + 1]
  std::vector<double> exponent;    // [nprim_pair] a + b
  std::vector<double> Px;          // [nprim_pair]
  std::vector<double> Py;
  std::vector<double> Pz;
  std::vector<double> prefactor;   // [nprim_pair]
  std::vector<double> alpha_a;     // [nprim_pair] a
  std::vector<double> alpha_b;     // [nprim_pair] b

  int npair() const { return static_cast<int>(ish.size()); }
  // View over host memory.
  ShellPairView view() const;
};

// Builds the pairs (i, j) of shells with l_i = la and l_j = lb; when la == lb
// only i >= j is kept. Primitive pairs with |prefactor| < prim_cutoff are
// dropped. Computes exact shell-pair Schwarz bounds and sorts pairs in
// descending `schwarz_bound` order.
ShellPairData build_shell_pairs(const Basis& basis, int la, int lb, double prim_cutoff = 0.0);

// Builds 1-center auxiliary shell pairs (P, unit) for RI 3-center (ab|P) and
// 2-center (P|Q) Coulomb integrals for shells in `aux_basis` with l == la.
ShellPairData build_aux_shell_pairs(const Basis& aux_basis, int la);

}  // namespace gpu4dft
