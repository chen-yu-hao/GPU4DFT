#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gpu4dft/one_electron.hpp"

namespace gpu4dft::dftd3 {

struct D3BJParams {
  double s6 = 1.0;
  double a1 = 0.0;
  double s8 = 0.0;
  double a2 = 0.0;  // Bohr
};

struct D3BJResult {
  double energy = 0.0;           // Hartree
  std::vector<double> gradient;  // [natom * 3] row-major (Hartree / Bohr)
  std::vector<double> cn;        // [natom] fractional coordination numbers
};

// Look up official Grimme DFT-D3(BJ) damping parameters (s6=1.0, atm=false).
// Normalizes names by lowercasing and stripping whitespace, '-', '_', '(', ')', '/'
// (and optional trailing "d3bj" / "d3" suffix). Throws std::invalid_argument if unsupported.
D3BJParams get_d3bj_params(std::string_view functional);

bool has_d3bj_params(std::string_view functional) noexcept;

std::vector<std::string> supported_d3bj_functionals();

// Compute DFT-D3(BJ) two-body dispersion energy and (optionally) analytical nuclear gradient.
// `coords_bohr` has length 3 * atomic_numbers.size() in Bohr.
D3BJResult compute_d3bj(const std::vector<int>& atomic_numbers,
                        const std::vector<double>& coords_bohr,
                        const std::string& functional,
                        bool compute_grad = true);

D3BJResult compute_d3bj_with_params(const std::vector<int>& atomic_numbers,
                                    const std::vector<double>& coords_bohr,
                                    const D3BJParams& params,
                                    bool compute_grad = true);

D3BJResult compute_d3bj(std::span<const Atom> atoms,
                        std::string_view functional,
                        bool compute_grad = true);

}  // namespace gpu4dft::dftd3
