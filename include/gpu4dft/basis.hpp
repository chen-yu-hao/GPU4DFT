#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace gpu4dft {

// One contracted shell of real solid-harmonic Gaussians.
//
// On input `coefficients` follow the basis-set-file convention: they multiply
// normalised primitives (this is what PySCF's Mole.bas_ctr_coeff returns).
// Basis's constructor rewrites them so that they multiply the raw radial
// primitives r^l exp(-a r^2) and the contracted radial function has unit norm,
// i.e. \int R(r)^2 r^2 dr = 1. The full basis function is R(r) Y_lm.
struct Shell {
  int l = 0;
  std::array<double, 3> center{};  // Bohr
  std::vector<double> exponents;
  std::vector<double> coefficients;
};

class Basis {
 public:
  // Validates and normalises the shells. Throws std::invalid_argument.
  explicit Basis(std::vector<Shell> shells);

  std::size_t nshell() const { return shells_.size(); }
  const Shell& shell(std::size_t i) const { return shells_.at(i); }
  const std::vector<Shell>& shells() const { return shells_; }

  // Number of spherical AOs and the first AO of each shell. AOs are ordered by
  // shell, then by m, matching PySCF's spherical ordering.
  int nao() const { return nao_; }
  int ao_offset(std::size_t ishell) const { return ao_offset_.at(ishell); }
  int max_l() const { return max_l_; }

 private:
  std::vector<Shell> shells_;
  std::vector<int> ao_offset_;
  int nao_ = 0;
  int max_l_ = 0;
};

}  // namespace gpu4dft
