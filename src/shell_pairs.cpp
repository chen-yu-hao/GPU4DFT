#include "gpu4dft/shell_pairs.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "gpu4dft/types.hpp"

namespace gpu4dft {
namespace {

constexpr double kPi = 3.14159265358979323846;
// Real solid harmonic Y_00 = 1 / sqrt(4 pi).
constexpr double kY00 = 0.28209479177387814347;

void append_pair(ShellPairData& pairs, const Basis& basis, int i, int j, double prim_cutoff) {
  const Shell& si = basis.shell(i);
  const Shell& sj = basis.shell(j);
  const auto& A = si.center;
  const auto& B = sj.center;
  const double ABx = A[0] - B[0];
  const double ABy = A[1] - B[1];
  const double ABz = A[2] - B[2];
  const double AB2 = ABx * ABx + ABy * ABy + ABz * ABz;

  for (std::size_t ka = 0; ka < si.exponents.size(); ++ka) {
    const double a = si.exponents[ka];
    for (std::size_t kb = 0; kb < sj.exponents.size(); ++kb) {
      const double b = sj.exponents[kb];
      const double p = a + b;
      const double prefactor = si.coefficients[ka] * sj.coefficients[kb] * kY00 * kY00 *
                               std::exp(-a * b / p * AB2) * std::pow(kPi / p, 1.5);
      if (std::abs(prefactor) < prim_cutoff) continue;
      pairs.exponent.push_back(p);
      pairs.Px.push_back((a * A[0] + b * B[0]) / p);
      pairs.Py.push_back((a * A[1] + b * B[1]) / p);
      pairs.Pz.push_back((a * A[2] + b * B[2]) / p);
      pairs.prefactor.push_back(prefactor);
    }
  }
  pairs.ish.push_back(i);
  pairs.jsh.push_back(j);
  pairs.prim_offset.push_back(static_cast<int>(pairs.exponent.size()));
}

}  // namespace

ShellPairView ShellPairData::view() const {
  return ShellPairView{npair(),   prim_offset.data(), exponent.data(), Px.data(),
                       Py.data(), Pz.data(),          prefactor.data()};
}

ShellPairData build_shell_pairs(const Basis& basis, int la, int lb, double prim_cutoff) {
  if (la < 0 || lb < 0) throw std::invalid_argument("angular momenta must be non-negative");
  if (la != 0 || lb != 0) {
    throw NotImplemented("shell pairs (l=" + std::to_string(la) + ", l=" + std::to_string(lb) +
                         ") are not implemented yet; only (s, s) is");
  }

  ShellPairData pairs;
  pairs.la = la;
  pairs.lb = lb;
  const int nshell = static_cast<int>(basis.nshell());
  for (int i = 0; i < nshell; ++i) {
    if (basis.shell(i).l != la) continue;
    const int jend = (la == lb) ? i + 1 : nshell;
    for (int j = 0; j < jend; ++j) {
      if (basis.shell(j).l != lb) continue;
      append_pair(pairs, basis, i, j, prim_cutoff);
    }
  }
  return pairs;
}

}  // namespace gpu4dft
