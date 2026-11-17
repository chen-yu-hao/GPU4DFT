#include "gpu4dft/shell_pairs.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/generated/eri_general.hpp"
#include "gpu4dft/harmonics.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kY00 = 0.28209479177387814347;  // 1 / sqrt(4 pi)
constexpr int kMaxSupportedL = 4;

void append_raw_pair(ShellPairData& pairs, const Basis& basis, int i, int j, double prim_cutoff) {
  const Shell& si = basis.shell(i);
  const Shell& sj = basis.shell(j);
  const auto& A = si.center;
  const auto& B = sj.center;
  const double ABx = A[0] - B[0];
  const double ABy = A[1] - B[1];
  const double ABz = A[2] - B[2];
  const double AB2 = ABx * ABx + ABy * ABy + ABz * ABz;

  const std::size_t start_prim = pairs.exponent.size();
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
      pairs.alpha_a.push_back(a);
      pairs.alpha_b.push_back(b);
    }
  }
  if (prim_cutoff > 0.0 && pairs.exponent.size() == start_prim) {
    return;
  }

  pairs.ish.push_back(i);
  pairs.jsh.push_back(j);
  pairs.ao_i.push_back(basis.ao_offset(i));
  pairs.ao_j.push_back(basis.ao_offset(j));
  pairs.Ax.push_back(A[0]);
  pairs.Ay.push_back(A[1]);
  pairs.Az.push_back(A[2]);
  pairs.Bx.push_back(B[0]);
  pairs.By.push_back(B[1]);
  pairs.Bz.push_back(B[2]);
  pairs.prim_offset.push_back(static_cast<int>(pairs.exponent.size()));
}

void compute_schwarz_and_sort(ShellPairData& raw, bool sort_descending) {
  const int np = raw.npair();
  raw.schwarz_bound.assign(np, 0.0);
  if (np == 0) return;

  const int la = raw.la;
  const int lb = raw.lb;
  const int nsa = n_sph(la);
  const int nsb = n_sph(lb);
  const int nsab = nsa * nsb;
  const ShellPairView v = raw.view();

  #if defined(_OPENMP)
  #pragma omp parallel
  #endif
  {
    std::vector<double> diag_block(static_cast<std::size_t>(nsab) * nsab, 0.0);
    #if defined(_OPENMP)
    #pragma omp for schedule(dynamic, 16)
    #endif
    for (int P = 0; P < np; ++P) {
      generated::eri_general_eval<double>(la, lb, la, lb, v, P, v, P, diag_block.data());
      double max_diag = 0.0;
      for (int uv = 0; uv < nsab; ++uv) {
        const double val = std::abs(diag_block[uv * nsab + uv]);
        if (val > max_diag) max_diag = val;
      }
      raw.schwarz_bound[P] = std::sqrt(max_diag);
    }
  }

  if (!sort_descending || np <= 1) return;

  std::vector<int> order(np);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return raw.schwarz_bound[a] > raw.schwarz_bound[b];
  });

  ShellPairData sorted;
  sorted.la = raw.la;
  sorted.lb = raw.lb;
  sorted.ish.reserve(np);
  sorted.jsh.reserve(np);
  sorted.ao_i.reserve(np);
  sorted.ao_j.reserve(np);
  sorted.Ax.reserve(np);
  sorted.Ay.reserve(np);
  sorted.Az.reserve(np);
  sorted.Bx.reserve(np);
  sorted.By.reserve(np);
  sorted.Bz.reserve(np);
  sorted.schwarz_bound.reserve(np);
  sorted.prim_offset.reserve(np + 1);
  sorted.exponent.reserve(raw.exponent.size());
  sorted.Px.reserve(raw.Px.size());
  sorted.Py.reserve(raw.Py.size());
  sorted.Pz.reserve(raw.Pz.size());
  sorted.prefactor.reserve(raw.prefactor.size());
  sorted.alpha_a.reserve(raw.alpha_a.size());
  sorted.alpha_b.reserve(raw.alpha_b.size());

  for (int idx = 0; idx < np; ++idx) {
    const int P = order[idx];
    sorted.ish.push_back(raw.ish[P]);
    sorted.jsh.push_back(raw.jsh[P]);
    sorted.ao_i.push_back(raw.ao_i[P]);
    sorted.ao_j.push_back(raw.ao_j[P]);
    sorted.Ax.push_back(raw.Ax[P]);
    sorted.Ay.push_back(raw.Ay[P]);
    sorted.Az.push_back(raw.Az[P]);
    sorted.Bx.push_back(raw.Bx[P]);
    sorted.By.push_back(raw.By[P]);
    sorted.Bz.push_back(raw.Bz[P]);
    sorted.schwarz_bound.push_back(raw.schwarz_bound[P]);

    const int pb = raw.prim_offset[P];
    const int pe = raw.prim_offset[P + 1];
    for (int p = pb; p < pe; ++p) {
      sorted.exponent.push_back(raw.exponent[p]);
      sorted.Px.push_back(raw.Px[p]);
      sorted.Py.push_back(raw.Py[p]);
      sorted.Pz.push_back(raw.Pz[p]);
      sorted.prefactor.push_back(raw.prefactor[p]);
      sorted.alpha_a.push_back(raw.alpha_a[p]);
      sorted.alpha_b.push_back(raw.alpha_b[p]);
    }
    sorted.prim_offset.push_back(static_cast<int>(sorted.exponent.size()));
  }

  raw = std::move(sorted);
}

}  // namespace

ShellPairView ShellPairData::view() const {
  return ShellPairView{
      npair(),
      prim_offset.data(),
      exponent.data(),
      Px.data(),
      Py.data(),
      Pz.data(),
      prefactor.data(),
      Ax.data(),
      Ay.data(),
      Az.data(),
      Bx.data(),
      By.data(),
      Bz.data(),
      alpha_a.data(),
      alpha_b.data(),
      schwarz_bound.data(),
      ish.data(),
      jsh.data(),
      ao_i.data(),
      ao_j.data(),
  };
}

ShellPairData build_shell_pairs(const Basis& basis, int la, int lb, double prim_cutoff) {
  if (la < 0 || lb < 0) throw std::invalid_argument("angular momenta must be non-negative");
  if (la > kMaxSupportedL || lb > kMaxSupportedL) {
    throw NotImplemented("shell pairs (l=" + std::to_string(la) + ", l=" + std::to_string(lb) +
                         ") exceed maximum supported angular momentum l=" +
                         std::to_string(kMaxSupportedL));
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
      append_raw_pair(pairs, basis, i, j, prim_cutoff);
    }
  }

  compute_schwarz_and_sort(pairs, /*sort_descending=*/true);
  return pairs;
}

ShellPairData build_aux_shell_pairs(const Basis& aux_basis, int la) {
  if (la < 0) throw std::invalid_argument("auxiliary angular momentum must be non-negative");
  if (la > kMaxSupportedL) {
    throw NotImplemented("auxiliary shell l=" + std::to_string(la) +
                         " exceeds maximum supported l=" + std::to_string(kMaxSupportedL));
  }

  ShellPairData pairs;
  pairs.la = la;
  pairs.lb = 0;
  const int nshell = static_cast<int>(aux_basis.nshell());
  for (int i = 0; i < nshell; ++i) {
    const Shell& s = aux_basis.shell(i);
    if (s.l != la) continue;
    const auto& A = s.center;
    for (std::size_t k = 0; k < s.exponents.size(); ++k) {
      const double a = s.exponents[k];
      const double prefactor = s.coefficients[k] * kY00 * std::pow(kPi / a, 1.5);
      pairs.exponent.push_back(a);
      pairs.Px.push_back(A[0]);
      pairs.Py.push_back(A[1]);
      pairs.Pz.push_back(A[2]);
      pairs.prefactor.push_back(prefactor);
      pairs.alpha_a.push_back(a);
      pairs.alpha_b.push_back(0.0);
    }
    pairs.ish.push_back(i);
    pairs.jsh.push_back(i);
    pairs.ao_i.push_back(aux_basis.ao_offset(i));
    pairs.ao_j.push_back(0);
    pairs.Ax.push_back(A[0]);
    pairs.Ay.push_back(A[1]);
    pairs.Az.push_back(A[2]);
    pairs.Bx.push_back(A[0]);
    pairs.By.push_back(A[1]);
    pairs.Bz.push_back(A[2]);
    pairs.prim_offset.push_back(static_cast<int>(pairs.exponent.size()));
  }

  compute_schwarz_and_sort(pairs, /*sort_descending=*/false);
  return pairs;
}

}  // namespace gpu4dft
