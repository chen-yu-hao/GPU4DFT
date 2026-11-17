#include "gpu4dft/response.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu4dft/eri.hpp"
#include "gpu4dft/generated/xc_kernels.hpp"

#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define GPU4DFT_RESP_C_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define GPU4DFT_RESP_C_EXPORT extern "C"
#endif

namespace gpu4dft {

const char* to_string(ResponseMethod method) {
  switch (method) {
    case ResponseMethod::TDA:
      return "tda";
    case ResponseMethod::TDDFT:
      return "tddft";
  }
  return "unknown";
}

namespace {

constexpr int kMaxSupportedDipoleL = 4;
// Exact NIST / PySCF (`pyscf.data.nist`) physical constants:
constexpr double kHartreeToEv = 27.21138602;
constexpr double kHartreeToCm1 = 219474.63111558527;
constexpr double kAmuToElectronMass = 1822.8884858012984;

using SteadyClock = std::chrono::steady_clock;

double elapsed_sec(SteadyClock::time_point t0, SteadyClock::time_point t1) {
  return std::chrono::duration<double>(t1 - t0).count();
}

bool is_hf_xc_name(std::string_view raw) {
  std::string norm;
  norm.reserve(raw.size());
  for (char ch : raw) {
    if (std::isspace(static_cast<unsigned char>(ch)) != 0 || ch == '-' || ch == '_') continue;
    norm.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return norm.empty() || norm == "hf" || norm == "rhf" || norm == "cis" || norm == "tdhf";
}

double dot_vec(std::size_t n, const double* a, const double* b) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

double norm2_vec(std::size_t n, const double* a) {
  return std::sqrt(dot_vec(n, a, a));
}

// ===========================================================================
// Basis Packing for Dipole Integrals
// ===========================================================================

struct HostDipoleBasisTable {
  std::vector<int> shell_l;
  std::vector<int> shell_ao;
  std::vector<int> prim_offset{0};
  std::vector<double> shell_center;
  std::vector<double> prim_exp;
  std::vector<double> prim_coeff;
  int nao = 0;

  detail::BasisDeviceView view() const {
    return detail::BasisDeviceView{
        static_cast<int>(shell_l.size()),
        nao,
        0,
        shell_l.data(),
        shell_ao.data(),
        nullptr,
        prim_offset.data(),
        shell_center.data(),
        prim_exp.data(),
        prim_coeff.data(),
        nullptr,
        nullptr,
    };
  }
};

HostDipoleBasisTable pack_dipole_basis(const Basis& basis) {
  if (basis.max_l() > kMaxSupportedDipoleL) {
    throw NotImplemented("dipole_matrices supports up to g shells (l <= 4); basis has l = " +
                         std::to_string(basis.max_l()));
  }
  HostDipoleBasisTable t;
  t.nao = basis.nao();
  const std::size_t nsh = basis.nshell();
  t.shell_l.reserve(nsh);
  t.shell_ao.reserve(nsh);
  t.shell_center.reserve(nsh * 3);
  t.prim_offset.reserve(nsh + 1);

  for (std::size_t i = 0; i < nsh; ++i) {
    const Shell& s = basis.shell(i);
    t.shell_l.push_back(s.l);
    t.shell_ao.push_back(basis.ao_offset(i));
    t.shell_center.push_back(s.center[0]);
    t.shell_center.push_back(s.center[1]);
    t.shell_center.push_back(s.center[2]);
    for (std::size_t k = 0; k < s.exponents.size(); ++k) {
      t.prim_exp.push_back(s.exponents[k]);
      t.prim_coeff.push_back(s.coefficients[k]);
    }
    t.prim_offset.push_back(static_cast<int>(t.prim_exp.size()));
  }
  return t;
}

// ===========================================================================
// Host Non-Symmetric Direct JK Helper
// ===========================================================================

struct HostDirectResponseJK {
  Basis basis;
  std::vector<ShellPairData> pair_lists;
  std::vector<int> shell_cart_offset;
  std::vector<std::vector<double>> c_mats;
  int ncart_total = 0;
  double schwarz_cutoff = 1e-12;

  HostDirectResponseJK(const Basis& b, double cutoff) : basis(b), schwarz_cutoff(cutoff) {
    if (basis.max_l() > 4) {
      throw NotImplemented("Direct response JK supports up to g shells (l <= 4)");
    }
    const int nsh = static_cast<int>(basis.nshell());
    shell_cart_offset.reserve(nsh);
    c_mats.resize(nsh);
    for (int s = 0; s < nsh; ++s) {
      shell_cart_offset.push_back(ncart_total);
      const int l = basis.shell(s).l;
      const int nc = n_cart(l);
      const int ns = n_sph(l);
      ncart_total += nc;
      c_mats[s].assign(static_cast<std::size_t>(ns) * nc, 0.0);
      double unit[15] = {0.0};
      double col[9] = {0.0};
      for (int ic = 0; ic < nc; ++ic) {
        unit[ic] = 1.0;
        harmonics::cart_to_sph_axis<double>(l, unit, 1, col, 1);
        for (int m = 0; m < ns; ++m) c_mats[s][m * nc + ic] = col[m];
        unit[ic] = 0.0;
      }
    }

    const int max_l = basis.max_l();
    for (int la = 0; la <= max_l; ++la) {
      for (int lb = 0; lb <= la; ++lb) {
        ShellPairData pd = build_shell_pairs(basis, la, lb, 1e-14);
        if (pd.npair() > 0) pair_lists.push_back(std::move(pd));
      }
    }
  }

  void eval(const double* dm, double* vj_out, double* vk_out) const {
    const int nao = basis.nao();
    const int nsh = static_cast<int>(basis.nshell());
    const bool do_j = (vj_out != nullptr);
    const bool do_k = (vk_out != nullptr);
    const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
    if (do_j) std::fill_n(vj_out, nao2, 0.0);
    if (do_k) std::fill_n(vk_out, nao2, 0.0);
    if ((!do_j && !do_k) || nao == 0) return;

    // 1. Transform non-symmetric spherical dm -> Cartesian d_cart (NO symmetrization!)
    const std::size_t nc2 = static_cast<std::size_t>(ncart_total) * ncart_total;
    std::vector<double> d_cart(nc2, 0.0);
    std::vector<double> d_shell_max(static_cast<std::size_t>(nsh) * nsh, 0.0);
    double global_dmax = 0.0;

    for (int i = 0; i < nsh; ++i) {
      const int li = basis.shell(i).l;
      const int nci = n_cart(li), nsi = n_sph(li);
      const int aoi = basis.ao_offset(i);
      const int ci0 = shell_cart_offset[i];
      const auto& Ci = c_mats[i];

      for (int j = 0; j < nsh; ++j) {
        const int lj = basis.shell(j).l;
        const int ncj = n_cart(lj), nsj = n_sph(lj);
        const int aoj = basis.ao_offset(j);
        const int cj0 = shell_cart_offset[j];
        const auto& Cj = c_mats[j];

        double blk_max = 0.0;
        double tmp[9 * 15] = {0.0};
        for (int mi = 0; mi < nsi; ++mi) {
          for (int mj = 0; mj < nsj; ++mj) {
            const int u = aoi + mi;
            const int v = aoj + mj;
            const double d_uv = dm[u * nao + v];
            const double a_sym = std::max(std::abs(d_uv), std::abs(dm[v * nao + u]));
            if (a_sym > blk_max) blk_max = a_sym;
            for (int cj = 0; cj < ncj; ++cj) {
              tmp[mi * ncj + cj] += d_uv * Cj[mj * ncj + cj];
            }
          }
        }
        d_shell_max[i * nsh + j] = blk_max;
        if (blk_max > global_dmax) global_dmax = blk_max;

        for (int ci = 0; ci < nci; ++ci) {
          for (int cj = 0; cj < ncj; ++cj) {
            double acc = 0.0;
            for (int mi = 0; mi < nsi; ++mi) acc += Ci[mi * nci + ci] * tmp[mi * ncj + cj];
            d_cart[(ci0 + ci) * ncart_total + (cj0 + cj)] = acc;
          }
        }
      }
    }

    const double global_scale = (do_j ? 4.0 : 1.0) * global_dmax;
    if (global_scale == 0.0) return;

    // 2. Build Schwarz-screened task tiles
    constexpr int kTile = 8;
    std::vector<detail::RespJKTaskTile> tasks;
    for (int bp = 0; bp < static_cast<int>(pair_lists.size()); ++bp) {
      const ShellPairData& bra = pair_lists[bp];
      for (int P = 0; P < bra.npair(); ++P) {
        const double q_ab = bra.schwarz_bound[P];
        if (q_ab * q_ab * global_scale < schwarz_cutoff * 1e-6) continue;
        for (int kp = 0; kp <= bp; ++kp) {
          const ShellPairData& ket = pair_lists[kp];
          const int q_lim = (bp == kp) ? (P + 1) : ket.npair();
          for (int Q0 = 0; Q0 < q_lim; Q0 += kTile) {
            if (q_ab * ket.schwarz_bound[Q0] * global_scale < schwarz_cutoff) break;
            tasks.push_back(detail::RespJKTaskTile{bp, P, kp, Q0, std::min(Q0 + kTile, q_lim)});
          }
        }
      }
    }
    if (tasks.empty()) return;

    std::vector<double> j_cart(do_j ? nc2 : 0, 0.0);
    std::vector<double> k_cart(do_k ? nc2 : 0, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
      std::vector<double> j_loc(do_j ? nc2 : 0, 0.0);
      std::vector<double> k_loc(do_k ? nc2 : 0, 0.0);
      auto accum_j = [&](int r, int c, double val) { j_loc[r * ncart_total + c] += val; };
      auto accum_k = [&](int r, int c, double val) { k_loc[r * ncart_total + c] += val; };

#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 4)
#endif
      for (std::ptrdiff_t tid = 0; tid < static_cast<std::ptrdiff_t>(tasks.size()); ++tid) {
        const auto& t = tasks[static_cast<std::size_t>(tid)];
        const ShellPairData& bra = pair_lists[t.bp];
        const ShellPairData& ket = pair_lists[t.kp];
        const ShellPairView bv = bra.view();
        const ShellPairView kv = ket.view();
        const int P = t.P;
        const int i = bra.ish[P], j = bra.jsh[P];
        const bool same_ij = (i == j);
        const double q_ab = bra.schwarz_bound[P];
        const int off_a = shell_cart_offset[i], off_b = shell_cart_offset[j];

        for (int Q = t.Q_begin; Q < t.Q_end; ++Q) {
          const double q_cd = ket.schwarz_bound[Q];
          if (q_ab * q_cd * global_scale < schwarz_cutoff) break;
          const int k = ket.ish[Q], l = ket.jsh[Q];
          double dmax = 0.0;
          if (do_j) {
            dmax = std::max(dmax, 4.0 * d_shell_max[i * nsh + j]);
            dmax = std::max(dmax, 4.0 * d_shell_max[k * nsh + l]);
          }
          if (do_k) {
            dmax = std::max(dmax, d_shell_max[i * nsh + k]);
            dmax = std::max(dmax, d_shell_max[i * nsh + l]);
            dmax = std::max(dmax, d_shell_max[j * nsh + k]);
            dmax = std::max(dmax, d_shell_max[j * nsh + l]);
          }
          if (q_ab * q_cd * dmax < schwarz_cutoff) continue;

          const bool same_kl = (k == l);
          const bool same_PQ = (t.bp == t.kp) && (P == Q);
          const int off_c = shell_cart_offset[k], off_d = shell_cart_offset[l];

          detail::eval_nonsym_jk_quartet(bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_ij,
                                         same_kl, same_PQ, do_j, do_k, d_cart.data(), ncart_total,
                                         off_a, off_b, off_c, off_d, accum_j, accum_k);
        }
      }

#if defined(_OPENMP)
#pragma omp critical
#endif
      {
        if (do_j) {
          for (std::size_t q = 0; q < nc2; ++q) j_cart[q] += j_loc[q];
        }
        if (do_k) {
          for (std::size_t q = 0; q < nc2; ++q) k_cart[q] += k_loc[q];
        }
      }
    }

    // 3. Transform Cartesian -> Spherical (J symmetric, K NON-symmetric!)
    auto cart_to_sph_matrix = [&](const std::vector<double>& m_cart, double* m_sph, bool sym) {
      for (int i = 0; i < nsh; ++i) {
        const int li = basis.shell(i).l;
        const int nci = n_cart(li), nsi = n_sph(li);
        const int aoi = basis.ao_offset(i);
        const int ci0 = shell_cart_offset[i];
        const int j_end = sym ? (i + 1) : nsh;
        for (int j = 0; j < j_end; ++j) {
          const int lj = basis.shell(j).l;
          const int ncj = n_cart(lj), nsj = n_sph(lj);
          const int aoj = basis.ao_offset(j);
          const int cj0 = shell_cart_offset[j];
          double blk_cart[225];
          for (int a = 0; a < nci; ++a) {
            for (int b = 0; b < ncj; ++b) {
              if (sym) {
                blk_cart[a * ncj + b] =
                    0.5 * (m_cart[(ci0 + a) * ncart_total + (cj0 + b)] +
                           m_cart[(cj0 + b) * ncart_total + (ci0 + a)]);
              } else {
                blk_cart[a * ncj + b] = m_cart[(ci0 + a) * ncart_total + (cj0 + b)];
              }
            }
          }
          double blk_sph[81];
          harmonics::cart_to_sph_2d<double>(li, lj, blk_cart, blk_sph);
          for (int mi = 0; mi < nsi; ++mi) {
            for (int mj = 0; mj < nsj; ++mj) {
              const double v = blk_sph[mi * nsj + mj];
              m_sph[(aoi + mi) * nao + (aoj + mj)] = v;
              if (sym) m_sph[(aoj + mj) * nao + (aoi + mi)] = v;
            }
          }
        }
      }
    };

    if (do_j) cart_to_sph_matrix(j_cart, vj_out, true);
    if (do_k) cart_to_sph_matrix(k_cart, vk_out, false);
  }
};

// ===========================================================================
// Host Cached Second-Derivative XC Hessian (`HostFxcEngine`)
// ===========================================================================

struct HostFxcPoint {
  double wg = 0.0;
  double gx0 = 0.0, gy0 = 0.0, gz0 = 0.0;
  double vsigma = 0.0;
  double v2rho2 = 0.0, v2rhosigma = 0.0, v2sigma2 = 0.0;
  double v2rhotau = 0.0, v2sigmatau = 0.0, v2tau2 = 0.0;
};

struct HostFxcBatchCache {
  int na = 0;
  int ng = 0;
  std::vector<int> active_aos;
  std::vector<double> phi;  // [ncomp * na * ng]
  std::vector<HostFxcPoint> pts;
};

struct HostFxcEngine {
  int nao = 0;
  int xc_family = 0;
  int ncomp = 1;
  std::vector<HostFxcBatchCache> batches;

  HostFxcEngine(const detail::FlatBasisData& flat, const MolecularGrid& grid, int func_id,
                int family, const double* dm0_host, double rho_cutoff)
      : nao(flat.nao), xc_family(family), ncomp((family == 0) ? 1 : 4) {
    const detail::FlatBasisView bview = flat.view();
    const auto fid = static_cast<xc_gen::XcFunctionalId>(func_id);
    const int deriv = (xc_family == 0) ? 0 : 1;
    const auto& gbatches = grid.batches();
    const auto& coords = grid.coords();
    const auto& weights = grid.weights();

    for (const SpatialBatch& b : gbatches) {
      const int ng = static_cast<int>(b.npoints);
      const int na = static_cast<int>(b.active_aos.size());
      if (ng == 0 || na == 0) continue;

      HostFxcBatchCache cb;
      cb.na = na;
      cb.ng = ng;
      cb.active_aos = b.active_aos;
      const std::size_t comp_stride = static_cast<std::size_t>(na) * ng;
      cb.phi.assign(static_cast<std::size_t>(ncomp) * comp_stride, 0.0);
      cb.pts.resize(static_cast<std::size_t>(ng));

      std::size_t cursor = 0;
      for (int is : b.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (int g = 0; g < ng; ++g) {
          const std::size_t gp = b.point_offset + static_cast<std::size_t>(g);
          detail::eval_shell_at_point(bview, is, coords[3 * gp + 0], coords[3 * gp + 1],
                                      coords[3 * gp + 2], deriv, cb.phi.data() + cursor * ng + g,
                                      comp_stride, static_cast<std::size_t>(ng));
        }
        cursor += static_cast<std::size_t>(nm);
      }

      std::vector<double> d0_act(static_cast<std::size_t>(na) * na, 0.0);
      for (int u = 0; u < na; ++u) {
        const int gu = b.active_aos[static_cast<std::size_t>(u)];
        for (int v = 0; v < na; ++v) {
          const int gv = b.active_aos[static_cast<std::size_t>(v)];
          d0_act[static_cast<std::size_t>(u) * na + v] =
              0.5 * (dm0_host[gu * nao + gv] + dm0_host[gv * nao + gu]);
        }
      }

      const int n_tcomp = (xc_family == 2) ? 4 : 1;
      std::vector<double> t0_mat(static_cast<std::size_t>(n_tcomp) * comp_stride, 0.0);
      for (int c = 0; c < n_tcomp; ++c) {
        const double* phic = cb.phi.data() + static_cast<std::size_t>(c) * comp_stride;
        double* tc = t0_mat.data() + static_cast<std::size_t>(c) * comp_stride;
        for (int u = 0; u < na; ++u) {
          for (int k = 0; k < na; ++k) {
            const double d_uk = d0_act[static_cast<std::size_t>(u) * na + k];
            if (d_uk == 0.0) continue;
            for (int g = 0; g < ng; ++g) {
              tc[static_cast<std::size_t>(u) * ng + g] +=
                  d_uk * phic[static_cast<std::size_t>(k) * ng + g];
            }
          }
        }
      }

      const double* phi0 = cb.phi.data();
      const double* phix = (xc_family >= 1) ? (cb.phi.data() + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family >= 1) ? (cb.phi.data() + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family >= 1) ? (cb.phi.data() + 3 * comp_stride) : nullptr;

      for (int g = 0; g < ng; ++g) {
        double rho0 = 0.0, gx0 = 0.0, gy0 = 0.0, gz0 = 0.0, tau0 = 0.0;
        for (int u = 0; u < na; ++u) {
          const std::size_t ug = static_cast<std::size_t>(u) * ng + g;
          const double t_ug = t0_mat[ug];
          rho0 += phi0[ug] * t_ug;
          if (xc_family >= 1) {
            gx0 += 2.0 * phix[ug] * t_ug;
            gy0 += 2.0 * phiy[ug] * t_ug;
            gz0 += 2.0 * phiz[ug] * t_ug;
          }
          if (xc_family == 2) {
            tau0 += 0.5 * (phix[ug] * t0_mat[1 * comp_stride + ug] +
                           phiy[ug] * t0_mat[2 * comp_stride + ug] +
                           phiz[ug] * t0_mat[3 * comp_stride + ug]);
          }
        }
        const double sig0 = (xc_family >= 1) ? (gx0 * gx0 + gy0 * gy0 + gz0 * gz0) : 0.0;
        xc_gen::XcPointOut pt_out;
        xc_gen::eval_xc_point(fid, rho0, sig0, tau0, 2, pt_out, rho_cutoff);

        HostFxcPoint hp;
        hp.wg = weights[b.point_offset + static_cast<std::size_t>(g)];
        hp.gx0 = gx0;
        hp.gy0 = gy0;
        hp.gz0 = gz0;
        hp.vsigma = pt_out.vsigma;
        hp.v2rho2 = pt_out.v2rho2;
        hp.v2rhosigma = pt_out.v2rhosigma;
        hp.v2sigma2 = pt_out.v2sigma2;
        hp.v2rhotau = pt_out.v2rhotau;
        hp.v2sigmatau = pt_out.v2sigmatau;
        hp.v2tau2 = pt_out.v2tau2;
        cb.pts[static_cast<std::size_t>(g)] = hp;
      }

      batches.push_back(std::move(cb));
    }
  }

  void eval(const double* sym_dm, double* fxc_out) const {
    const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
    std::fill_n(fxc_out, nao2, 0.0);
    if (nao == 0 || batches.empty()) return;

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
      std::vector<double> local_fxc(nao2, 0.0);
      std::vector<double> d_act, t_mat, w_mat, f_half, wvtau;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
      for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
        const HostFxcBatchCache& b = batches[static_cast<std::size_t>(ib)];
        const int na = b.na;
        const int ng = b.ng;
        const std::size_t comp_stride = static_cast<std::size_t>(na) * ng;

        d_act.resize(static_cast<std::size_t>(na) * na);
        for (int u = 0; u < na; ++u) {
          const int gu = b.active_aos[static_cast<std::size_t>(u)];
          for (int v = 0; v < na; ++v) {
            const int gv = b.active_aos[static_cast<std::size_t>(v)];
            d_act[static_cast<std::size_t>(u) * na + v] =
                0.5 * (sym_dm[gu * nao + gv] + sym_dm[gv * nao + gu]);
          }
        }

        const int n_tcomp = (xc_family == 2) ? 4 : 1;
        t_mat.assign(static_cast<std::size_t>(n_tcomp) * comp_stride, 0.0);
        for (int c = 0; c < n_tcomp; ++c) {
          const double* phic = b.phi.data() + static_cast<std::size_t>(c) * comp_stride;
          double* tc = t_mat.data() + static_cast<std::size_t>(c) * comp_stride;
          for (int u = 0; u < na; ++u) {
            for (int k = 0; k < na; ++k) {
              const double d_uk = d_act[static_cast<std::size_t>(u) * na + k];
              if (d_uk == 0.0) continue;
              for (int g = 0; g < ng; ++g) {
                tc[static_cast<std::size_t>(u) * ng + g] +=
                    d_uk * phic[static_cast<std::size_t>(k) * ng + g];
              }
            }
          }
        }

        const double* phi0 = b.phi.data();
        const double* phix = (xc_family >= 1) ? (b.phi.data() + 1 * comp_stride) : nullptr;
        const double* phiy = (xc_family >= 1) ? (b.phi.data() + 2 * comp_stride) : nullptr;
        const double* phiz = (xc_family >= 1) ? (b.phi.data() + 3 * comp_stride) : nullptr;

        w_mat.assign(comp_stride, 0.0);
        if (xc_family == 2) wvtau.assign(static_cast<std::size_t>(ng), 0.0);

        for (int g = 0; g < ng; ++g) {
          double drho = 0.0, dgx = 0.0, dgy = 0.0, dgz = 0.0, dtau = 0.0;
          for (int u = 0; u < na; ++u) {
            const std::size_t ug = static_cast<std::size_t>(u) * ng + g;
            const double t0_ug = t_mat[ug];
            drho += phi0[ug] * t0_ug;
            if (xc_family >= 1) {
              dgx += 2.0 * phix[ug] * t0_ug;
              dgy += 2.0 * phiy[ug] * t0_ug;
              dgz += 2.0 * phiz[ug] * t0_ug;
            }
            if (xc_family == 2) {
              dtau += 0.5 * (phix[ug] * t_mat[1 * comp_stride + ug] +
                             phiy[ug] * t_mat[2 * comp_stride + ug] +
                             phiz[ug] * t_mat[3 * comp_stride + ug]);
            }
          }

          const HostFxcPoint& p = b.pts[static_cast<std::size_t>(g)];
          const double wg = p.wg;
          const double dsigma =
              (xc_family >= 1) ? (2.0 * (p.gx0 * dgx + p.gy0 * dgy + p.gz0 * dgz)) : 0.0;

          const double dvrho = p.v2rho2 * drho + p.v2rhosigma * dsigma + p.v2rhotau * dtau;
          const double wv0 = 0.5 * wg * dvrho;

          double wvx = 0.0, wvy = 0.0, wvz = 0.0;
          if (xc_family >= 1) {
            const double dvsigma =
                p.v2rhosigma * drho + p.v2sigma2 * dsigma + p.v2sigmatau * dtau;
            wvx = 2.0 * wg * (dvsigma * p.gx0 + p.vsigma * dgx);
            wvy = 2.0 * wg * (dvsigma * p.gy0 + p.vsigma * dgy);
            wvz = 2.0 * wg * (dvsigma * p.gz0 + p.vsigma * dgz);
          }
          if (xc_family == 2) {
            const double dvtau = p.v2rhotau * drho + p.v2sigmatau * dsigma + p.v2tau2 * dtau;
            wvtau[static_cast<std::size_t>(g)] = 0.5 * wg * dvtau;
          }

          for (int u = 0; u < na; ++u) {
            const std::size_t ug = static_cast<std::size_t>(u) * ng + g;
            double val = wv0 * phi0[ug];
            if (xc_family >= 1) {
              val += wvx * phix[ug] + wvy * phiy[ug] + wvz * phiz[ug];
            }
            w_mat[ug] = val;
          }
        }

        f_half.assign(static_cast<std::size_t>(na) * na, 0.0);
        gemm(na, na, ng, 1.0, phi0, false, w_mat.data(), true, 0.0, f_half.data(), Backend::Host);

        if (xc_family == 2) {
          for (int c = 1; c <= 3; ++c) {
            const double* phib = b.phi.data() + static_cast<std::size_t>(c) * comp_stride;
            for (int u = 0; u < na; ++u) {
              for (int g = 0; g < ng; ++g) {
                w_mat[static_cast<std::size_t>(u) * ng + g] =
                    0.5 * wvtau[static_cast<std::size_t>(g)] *
                    phib[static_cast<std::size_t>(u) * ng + g];
              }
            }
            gemm(na, na, ng, 1.0, phib, false, w_mat.data(), true, 1.0, f_half.data(),
                 Backend::Host);
          }
        }

        for (int u = 0; u < na; ++u) {
          const int gu = b.active_aos[static_cast<std::size_t>(u)];
          for (int v = 0; v < na; ++v) {
            const int gv = b.active_aos[static_cast<std::size_t>(v)];
            local_fxc[static_cast<std::size_t>(gu) * nao + gv] +=
                f_half[static_cast<std::size_t>(u) * na + v] +
                f_half[static_cast<std::size_t>(v) * na + u];
          }
        }
      }

#if defined(_OPENMP)
#pragma omp critical
#endif
      {
        for (std::size_t q = 0; q < nao2; ++q) fxc_out[q] += local_fxc[q];
      }
    }
  }
};

// ===========================================================================
// Unified GPU & Host Orbital-Hessian Contraction Engine
// ===========================================================================

class OrbitalResponseOperator {
 public:
  OrbitalResponseOperator(int nao, int nmo, int n_occ, const double* mo_energies_host,
                          const double* mo_coeff_host, double alpha_x, Backend backend,
                          const ResponseCallbacks& callbacks, DeviceMemoryPool* ext_pool = nullptr)
      : nao_(nao),
        nmo_(nmo),
        n_occ_(n_occ),
        n_vir_(nmo - n_occ),
        n_ov_(n_occ * (nmo - n_occ)),
        alpha_x_(alpha_x),
        backend_(backend),
        cb_(callbacks) {
    if (!cb_.build_jk) {
      throw std::invalid_argument("OrbitalResponseOperator: callbacks.build_jk is required");
    }

    Co_host_.assign(static_cast<std::size_t>(nao_) * n_occ_, 0.0);
    Cv_host_.assign(static_cast<std::size_t>(nao_) * n_vir_, 0.0);
    for (int u = 0; u < nao_; ++u) {
      for (int i = 0; i < n_occ_; ++i) {
        Co_host_[static_cast<std::size_t>(u) * n_occ_ + i] =
            mo_coeff_host[static_cast<std::size_t>(u) * nmo_ + i];
      }
      for (int a = 0; a < n_vir_; ++a) {
        Cv_host_[static_cast<std::size_t>(u) * n_vir_ + a] =
            mo_coeff_host[static_cast<std::size_t>(u) * nmo_ + (n_occ_ + a)];
      }
    }

    delta_eps_.resize(static_cast<std::size_t>(n_ov_));
    sqrt_eps_.resize(static_cast<std::size_t>(n_ov_));
    inv_sqrt_eps_.resize(static_cast<std::size_t>(n_ov_));
    for (int i = 0; i < n_occ_; ++i) {
      for (int a = 0; a < n_vir_; ++a) {
        const std::size_t ia = static_cast<std::size_t>(i) * n_vir_ + a;
        const double de = mo_energies_host[n_occ_ + a] - mo_energies_host[i];
        delta_eps_[ia] = de;
        const double safe_de = std::max(de, 1e-12);
        sqrt_eps_[ia] = std::sqrt(safe_de);
        inv_sqrt_eps_[ia] = 1.0 / std::sqrt(safe_de);
      }
    }

    const std::size_t nao2 = static_cast<std::size_t>(nao_) * nao_;
    h_M_.assign(nao2, 0.0);
    h_sym_.assign(nao2, 0.0);
    h_J_.assign(nao2, 0.0);
    h_K_.assign(nao2, 0.0);
    h_Fxc_.assign(nao2, 0.0);
    h_Vp_.assign(nao2, 0.0);
    h_Vm_.assign(nao2, 0.0);
    h_tmp_uv_.assign(static_cast<std::size_t>(nao_) * n_vir_, 0.0);

#if defined(GPU4DFT_HAVE_CUDA)
    if (backend_ == Backend::Cuda) {
      DeviceMemoryPool& pool = (ext_pool != nullptr) ? *ext_pool : default_device_pool();
      ctx_ = std::make_unique<GpuLinalgContext>(&pool);
      StreamHandle s = ctx_->stream();

      d_Co_ = PooledBuffer<double>(pool, Co_host_.size(), s);
      d_Cv_ = PooledBuffer<double>(pool, Cv_host_.size(), s);
      d_delta_eps_ = PooledBuffer<double>(pool, delta_eps_.size(), s);
      d_sqrt_eps_ = PooledBuffer<double>(pool, sqrt_eps_.size(), s);
      d_x_ = PooledBuffer<double>(pool, static_cast<std::size_t>(n_ov_), s);
      d_z_ = PooledBuffer<double>(pool, static_cast<std::size_t>(n_ov_), s);
      d_sp_ = PooledBuffer<double>(pool, static_cast<std::size_t>(n_ov_), s);
      d_sm_ = PooledBuffer<double>(pool, static_cast<std::size_t>(n_ov_), s);
      d_tmp_uv_ = PooledBuffer<double>(pool, static_cast<std::size_t>(nao_) * n_vir_, s);
      d_M_ = PooledBuffer<double>(pool, nao2, s);
      d_sym_ = PooledBuffer<double>(pool, nao2, s);
      d_J_ = PooledBuffer<double>(pool, nao2, s);
      d_K_ = PooledBuffer<double>(pool, nao2, s);
      d_Fxc_ = PooledBuffer<double>(pool, nao2, s);
      d_Vp_ = PooledBuffer<double>(pool, nao2, s);
      d_Vm_ = PooledBuffer<double>(pool, nao2, s);

      copy_h2d_async(d_Co_.data(), Co_host_.data(), Co_host_.size() * sizeof(double), s);
      copy_h2d_async(d_Cv_.data(), Cv_host_.data(), Cv_host_.size() * sizeof(double), s);
      copy_h2d_async(d_delta_eps_.data(), delta_eps_.data(), delta_eps_.size() * sizeof(double), s);
      copy_h2d_async(d_sqrt_eps_.data(), sqrt_eps_.data(), sqrt_eps_.size() * sizeof(double), s);
      synchronize_stream(s);
    }
#else
    (void)ext_pool;
#endif
  }

  int nao() const noexcept { return nao_; }
  int n_occ() const noexcept { return n_occ_; }
  int n_vir() const noexcept { return n_vir_; }
  int n_ov() const noexcept { return n_ov_; }
  double alpha_x() const noexcept { return alpha_x_; }
  int matvec_count() const noexcept { return matvec_count_; }
  const std::vector<double>& delta_eps() const noexcept { return delta_eps_; }
  const std::vector<double>& sqrt_eps() const noexcept { return sqrt_eps_; }
  const std::vector<double>& inv_sqrt_eps() const noexcept { return inv_sqrt_eps_; }
  const std::vector<double>& Co_host() const noexcept { return Co_host_; }
  const std::vector<double>& Cv_host() const noexcept { return Cv_host_; }

  // Singlet TDA: y = A * x
  void apply_tda(const double* x_mo, double* y_mo, GemmMode mode, int ozaki_slices) {
    matvec_count_ += 1;
    const bool need_k = (std::abs(alpha_x_) > 0.0);
    const bool has_fxc = static_cast<bool>(cb_.build_fxc);

#if defined(GPU4DFT_HAVE_CUDA)
    if (backend_ == Backend::Cuda) {
      StreamHandle s = ctx_->stream();
      copy_h2d_async(d_x_.data(), x_mo, static_cast<std::size_t>(n_ov_) * sizeof(double), s);
      build_ao_transition_and_jk_fxc_cuda(d_x_.data(), need_k, has_fxc, mode, ozaki_slices, s);
      detail::assemble_tda_ao_potential_device(
          nao_, d_J_.data(), has_fxc ? d_Fxc_.data() : nullptr, alpha_x_,
          need_k ? d_K_.data() : nullptr, d_Vp_.data(), s);
      project_ao_to_mo_cuda(d_Vp_.data(), d_sp_.data(), mode, ozaki_slices, s);
      detail::add_diag_eps_mo_device(n_ov_, d_delta_eps_.data(), d_x_.data(), 1.0, d_sp_.data(), s);
      copy_d2h_async(y_mo, d_sp_.data(), static_cast<std::size_t>(n_ov_) * sizeof(double), s);
      synchronize_stream(s);
      return;
    }
#endif

    build_ao_transition_and_jk_fxc_host(x_mo, true, need_k, has_fxc, mode, ozaki_slices);
    const std::size_t nao2 = static_cast<std::size_t>(nao_) * nao_;
    for (std::size_t q = 0; q < nao2; ++q) {
      double v = 2.0 * h_J_[q];
      if (has_fxc) v += h_Fxc_[q];
      if (need_k) v -= alpha_x_ * h_K_[q];
      h_Vp_[q] = v;
    }
    project_ao_to_mo_host(h_Vp_.data(), y_mo, mode, ozaki_slices);
    for (int ia = 0; ia < n_ov_; ++ia) {
      y_mo[ia] += delta_eps_[static_cast<std::size_t>(ia)] * x_mo[ia];
    }
  }

  // Pure DFT Casida (\alpha_x == 0): y = \Omega * v = \Delta\eps^{1/2} (A + B) \Delta\eps^{1/2} v
  void apply_casida_omega(const double* v_mo, double* y_mo, GemmMode mode, int ozaki_slices) {
    matvec_count_ += 1;
    const bool has_fxc = static_cast<bool>(cb_.build_fxc);

#if defined(GPU4DFT_HAVE_CUDA)
    if (backend_ == Backend::Cuda) {
      StreamHandle s = ctx_->stream();
      copy_h2d_async(d_x_.data(), v_mo, static_cast<std::size_t>(n_ov_) * sizeof(double), s);
      detail::elemwise_mul_device(n_ov_, d_sqrt_eps_.data(), d_x_.data(), d_z_.data(), s);
      build_ao_transition_and_jk_fxc_cuda(d_z_.data(), false, has_fxc, mode, ozaki_slices, s);
      detail::assemble_rpa_ao_potentials_device(
          nao_, d_J_.data(), has_fxc ? d_Fxc_.data() : nullptr, 0.0, nullptr, d_Vp_.data(),
          nullptr, s);
      project_ao_to_mo_cuda(d_Vp_.data(), d_sp_.data(), mode, ozaki_slices, s);
      detail::add_diag_eps_mo_device(n_ov_, d_delta_eps_.data(), d_z_.data(), 1.0, d_sp_.data(), s);
      detail::elemwise_mul_device(n_ov_, d_sqrt_eps_.data(), d_sp_.data(), d_sm_.data(), s);
      copy_d2h_async(y_mo, d_sm_.data(), static_cast<std::size_t>(n_ov_) * sizeof(double), s);
      synchronize_stream(s);
      return;
    }
#endif

    std::vector<double> z(static_cast<std::size_t>(n_ov_));
    for (int ia = 0; ia < n_ov_; ++ia) {
      z[static_cast<std::size_t>(ia)] = sqrt_eps_[static_cast<std::size_t>(ia)] * v_mo[ia];
    }
    build_ao_transition_and_jk_fxc_host(z.data(), true, false, has_fxc, mode, ozaki_slices);
    const std::size_t nao2 = static_cast<std::size_t>(nao_) * nao_;
    for (std::size_t q = 0; q < nao2; ++q) {
      h_Vp_[q] = 4.0 * h_J_[q] + (has_fxc ? (2.0 * h_Fxc_[q]) : 0.0);
    }
    project_ao_to_mo_host(h_Vp_.data(), y_mo, mode, ozaki_slices);
    for (int ia = 0; ia < n_ov_; ++ia) {
      const double w =
          y_mo[ia] + delta_eps_[static_cast<std::size_t>(ia)] * z[static_cast<std::size_t>(ia)];
      y_mo[ia] = sqrt_eps_[static_cast<std::size_t>(ia)] * w;
    }
  }

  // Paired RPA / CPHF: simultaneously evaluates sp = (A + B) z (if sp != nullptr)
  // and sm = (A - B) z (if sm != nullptr) from ONE non-symmetric JK + F_xc call!
  void apply_rpa_pair(const double* z_mo, double* sp_out, double* sm_out, GemmMode mode,
                      int ozaki_slices) {
    matvec_count_ += 1;
    const bool need_plus = (sp_out != nullptr);
    const bool need_minus = (sm_out != nullptr);
    const bool need_j = need_plus;
    const bool need_k = (std::abs(alpha_x_) > 0.0) && (need_plus || need_minus);
    const bool has_fxc = need_plus && static_cast<bool>(cb_.build_fxc);

#if defined(GPU4DFT_HAVE_CUDA)
    if (backend_ == Backend::Cuda) {
      StreamHandle s = ctx_->stream();
      copy_h2d_async(d_x_.data(), z_mo, static_cast<std::size_t>(n_ov_) * sizeof(double), s);
      build_ao_transition_and_jk_fxc_cuda(d_x_.data(), need_k, has_fxc, mode, ozaki_slices, s,
                                          need_j);
      detail::assemble_rpa_ao_potentials_device(
          nao_, need_j ? d_J_.data() : nullptr, has_fxc ? d_Fxc_.data() : nullptr, alpha_x_,
          need_k ? d_K_.data() : nullptr, d_Vp_.data(), need_minus ? d_Vm_.data() : nullptr, s);
      if (need_plus) {
        project_ao_to_mo_cuda(d_Vp_.data(), d_sp_.data(), mode, ozaki_slices, s);
        detail::add_diag_eps_mo_device(n_ov_, d_delta_eps_.data(), d_x_.data(), 1.0, d_sp_.data(),
                                       s);
        copy_d2h_async(sp_out, d_sp_.data(), static_cast<std::size_t>(n_ov_) * sizeof(double), s);
      }
      if (need_minus) {
        if (need_k) {
          project_ao_to_mo_cuda(d_Vm_.data(), d_sm_.data(), mode, ozaki_slices, s);
          detail::add_diag_eps_mo_device(n_ov_, d_delta_eps_.data(), d_x_.data(), 1.0, d_sm_.data(),
                                         s);
        } else {
          detail::add_diag_eps_mo_device(n_ov_, d_delta_eps_.data(), d_x_.data(), 0.0, d_sm_.data(),
                                         s);
        }
        copy_d2h_async(sm_out, d_sm_.data(), static_cast<std::size_t>(n_ov_) * sizeof(double), s);
      }
      synchronize_stream(s);
      return;
    }
#endif

    build_ao_transition_and_jk_fxc_host(z_mo, need_j, need_k, has_fxc, mode, ozaki_slices);
    if (need_plus) {
      for (int u = 0; u < nao_; ++u) {
        for (int v = 0; v < nao_; ++v) {
          const std::size_t uv = static_cast<std::size_t>(u) * nao_ + v;
          const std::size_t vu = static_cast<std::size_t>(v) * nao_ + u;
          double vp = 4.0 * h_J_[uv];
          if (has_fxc) vp += 2.0 * h_Fxc_[uv];
          if (need_k) vp -= alpha_x_ * (h_K_[uv] + h_K_[vu]);
          h_Vp_[uv] = vp;
        }
      }
      project_ao_to_mo_host(h_Vp_.data(), sp_out, mode, ozaki_slices);
      for (int ia = 0; ia < n_ov_; ++ia) {
        sp_out[ia] += delta_eps_[static_cast<std::size_t>(ia)] * z_mo[ia];
      }
    }
    if (need_minus) {
      if (need_k) {
        for (int u = 0; u < nao_; ++u) {
          for (int v = 0; v < nao_; ++v) {
            const std::size_t uv = static_cast<std::size_t>(u) * nao_ + v;
            const std::size_t vu = static_cast<std::size_t>(v) * nao_ + u;
            h_Vm_[uv] = -alpha_x_ * (h_K_[uv] - h_K_[vu]);
          }
        }
        project_ao_to_mo_host(h_Vm_.data(), sm_out, mode, ozaki_slices);
        for (int ia = 0; ia < n_ov_; ++ia) {
          sm_out[ia] += delta_eps_[static_cast<std::size_t>(ia)] * z_mo[ia];
        }
      } else {
        for (int ia = 0; ia < n_ov_; ++ia) {
          sm_out[ia] = delta_eps_[static_cast<std::size_t>(ia)] * z_mo[ia];
        }
      }
    }
  }

 private:
  void build_ao_transition_and_jk_fxc_host(const double* x_mo, bool need_j, bool need_k,
                                           bool has_fxc, GemmMode mode, int ozaki_slices) {
    // M = C_o * X * C_v^T [nao x nao]
    gemm(nao_, n_vir_, n_occ_, 1.0, Co_host_.data(), false, x_mo, false, 0.0, h_tmp_uv_.data(),
         Backend::Host, mode, ozaki_slices);
    gemm(nao_, nao_, n_vir_, 1.0, h_tmp_uv_.data(), false, Cv_host_.data(), true, 0.0,
         h_M_.data(), Backend::Host, mode, ozaki_slices);

    if (need_j || need_k) {
      cb_.build_jk(h_M_.data(), need_j ? h_J_.data() : nullptr, need_k ? h_K_.data() : nullptr,
                   nullptr);
    }
    if (has_fxc) {
      for (int u = 0; u < nao_; ++u) {
        for (int v = 0; v < nao_; ++v) {
          h_sym_[static_cast<std::size_t>(u) * nao_ + v] =
              h_M_[static_cast<std::size_t>(u) * nao_ + v] +
              h_M_[static_cast<std::size_t>(v) * nao_ + u];
        }
      }
      cb_.build_fxc(h_sym_.data(), h_Fxc_.data(), nullptr);
    }
  }

  void project_ao_to_mo_host(const double* V_ao, double* out_mo, GemmMode mode, int ozaki_slices) {
    gemm(nao_, n_vir_, nao_, 1.0, V_ao, false, Cv_host_.data(), false, 0.0, h_tmp_uv_.data(),
         Backend::Host, mode, ozaki_slices);
    gemm(n_occ_, n_vir_, nao_, 1.0, Co_host_.data(), true, h_tmp_uv_.data(), false, 0.0, out_mo,
         Backend::Host, mode, ozaki_slices);
  }

#if defined(GPU4DFT_HAVE_CUDA)
  void build_ao_transition_and_jk_fxc_cuda(const double* d_x_in, bool need_k, bool has_fxc,
                                           GemmMode mode, int ozaki_slices, StreamHandle s,
                                           bool need_j = true) {
    const std::size_t nao2 = static_cast<std::size_t>(nao_) * nao_;
    gemm_device(*ctx_, nao_, n_vir_, n_occ_, 1.0, d_Co_.data(), n_occ_, false, d_x_in, n_vir_,
                false, 0.0, d_tmp_uv_.data(), n_vir_, mode, s, ozaki_slices);
    gemm_device(*ctx_, nao_, nao_, n_vir_, 1.0, d_tmp_uv_.data(), n_vir_, false, d_Cv_.data(),
                n_vir_, true, 0.0, d_M_.data(), nao_, mode, s, ozaki_slices);

    if (need_j || need_k) {
      if (cb_.device_resident) {
        cb_.build_jk(d_M_.data(), need_j ? d_J_.data() : nullptr, need_k ? d_K_.data() : nullptr,
                     reinterpret_cast<void*>(s));
      } else {
        copy_d2h_async(h_M_.data(), d_M_.data(), nao2 * sizeof(double), s);
        synchronize_stream(s);
        cb_.build_jk(h_M_.data(), need_j ? h_J_.data() : nullptr, need_k ? h_K_.data() : nullptr,
                     nullptr);
        if (need_j) copy_h2d_async(d_J_.data(), h_J_.data(), nao2 * sizeof(double), s);
        if (need_k) copy_h2d_async(d_K_.data(), h_K_.data(), nao2 * sizeof(double), s);
      }
    }

    if (has_fxc) {
      detail::form_sym_sum_transpose_device(nao_, d_M_.data(), d_sym_.data(), s);
      if (cb_.fxc_device_resident) {
        cb_.build_fxc(d_sym_.data(), d_Fxc_.data(), reinterpret_cast<void*>(s));
      } else {
        copy_d2h_async(h_sym_.data(), d_sym_.data(), nao2 * sizeof(double), s);
        synchronize_stream(s);
        cb_.build_fxc(h_sym_.data(), h_Fxc_.data(), nullptr);
        copy_h2d_async(d_Fxc_.data(), h_Fxc_.data(), nao2 * sizeof(double), s);
      }
    }
  }

  void project_ao_to_mo_cuda(const double* d_V_ao, double* d_out_mo, GemmMode mode,
                             int ozaki_slices, StreamHandle s) {
    gemm_device(*ctx_, nao_, n_vir_, nao_, 1.0, d_V_ao, nao_, false, d_Cv_.data(), n_vir_, false,
                0.0, d_tmp_uv_.data(), n_vir_, mode, s, ozaki_slices);
    gemm_device(*ctx_, n_occ_, n_vir_, nao_, 1.0, d_Co_.data(), n_occ_, true, d_tmp_uv_.data(),
                n_vir_, false, 0.0, d_out_mo, n_vir_, mode, s, ozaki_slices);
  }
#endif

  int nao_ = 0;
  int nmo_ = 0;
  int n_occ_ = 0;
  int n_vir_ = 0;
  int n_ov_ = 0;
  double alpha_x_ = 1.0;
  Backend backend_ = Backend::Host;
  ResponseCallbacks cb_;
  int matvec_count_ = 0;

  std::vector<double> Co_host_;
  std::vector<double> Cv_host_;
  std::vector<double> delta_eps_;
  std::vector<double> sqrt_eps_;
  std::vector<double> inv_sqrt_eps_;

  std::vector<double> h_M_, h_sym_, h_J_, h_K_, h_Fxc_, h_Vp_, h_Vm_, h_tmp_uv_;

#if defined(GPU4DFT_HAVE_CUDA)
  std::unique_ptr<GpuLinalgContext> ctx_;
  PooledBuffer<double> d_Co_, d_Cv_, d_delta_eps_, d_sqrt_eps_;
  PooledBuffer<double> d_x_, d_z_, d_sp_, d_sm_, d_tmp_uv_;
  PooledBuffer<double> d_M_, d_sym_, d_J_, d_K_, d_Fxc_, d_Vp_, d_Vm_;
#endif
};

// ===========================================================================
// Krylov Subspace Helpers (Two-Pass Modified Gram-Schmidt & Initial Guess)
// ===========================================================================

std::vector<int> select_initial_guess_indices(const std::vector<double>& diag_vals, int n_states) {
  const int n = static_cast<int>(diag_vals.size());
  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return diag_vals[static_cast<std::size_t>(a)] < diag_vals[static_cast<std::size_t>(b)];
  });

  int n_guess = std::min(n, std::max(3 * n_states + 6, 18));
  // Never split degenerate / near-degenerate unperturbed orbital transition shells!
  while (n_guess < n) {
    const double prev = diag_vals[static_cast<std::size_t>(order[n_guess - 1])];
    const double curr = diag_vals[static_cast<std::size_t>(order[n_guess])];
    if (std::abs(curr - prev) > 1e-5) break;
    ++n_guess;
  }
  order.resize(static_cast<std::size_t>(n_guess));
  return order;
}

// Appends `cand` [n_ov] into `basis_vecs` (each [n_ov]) after 2-pass MGS orthonormalization.
// Returns true if retained (post-projection norm > ortho_tol).
bool append_orthonormal_vector(int n_ov, std::vector<std::vector<double>>& basis_vecs,
                               std::vector<double> cand, double ortho_tol = 1e-7) {
  const std::size_t n = static_cast<std::size_t>(n_ov);
  const double init_norm = norm2_vec(n, cand.data());
  if (!(init_norm > 1e-14)) return false;
  for (std::size_t i = 0; i < n; ++i) cand[i] /= init_norm;

  for (int pass = 0; pass < 2; ++pass) {
    for (const auto& bv : basis_vecs) {
      const double proj = dot_vec(n, bv.data(), cand.data());
      for (std::size_t i = 0; i < n; ++i) cand[i] -= proj * bv[i];
    }
    const double nrm = norm2_vec(n, cand.data());
    if (!(nrm > ortho_tol)) return false;
    for (std::size_t i = 0; i < n; ++i) cand[i] /= nrm;
  }

  basis_vecs.push_back(std::move(cand));
  return true;
}

// Hermitian Block-Davidson solver used for both TDA (A X = \omega X) and
// Pure-DFT Casida (\Omega F = \omega^2 F).
struct HermitianDavidsonResult {
  bool converged = false;
  int iterations = 0;
  std::vector<double> eigenvalues;           // [n_states]
  std::vector<std::vector<double>> vectors;  // [n_states][n_ov]
  std::vector<double> residuals;             // [n_states]
  std::vector<DavidsonIterationLog> history;
};

HermitianDavidsonResult run_hermitian_davidson(
    int n_ov, int n_states, const std::vector<double>& diag_approx, const ResponseOptions& opts,
    const std::function<void(const double*, double*, GemmMode)>& matvec_fn) {
  HermitianDavidsonResult out;
  const int n_roots = std::min(n_states, n_ov);
  const int n_track = std::min(n_ov, std::max(n_roots + 3, n_roots));
  const int max_sub = std::max({opts.max_subspace, 6 * n_track, 24});

  const std::vector<int> guess_idx = select_initial_guess_indices(diag_approx, n_roots);
  std::vector<std::vector<double>> V_sub;
  std::vector<std::vector<double>> S_sub;
  V_sub.reserve(static_cast<std::size_t>(max_sub));
  S_sub.reserve(static_cast<std::size_t>(max_sub));

  for (int idx : guess_idx) {
    std::vector<double> e(static_cast<std::size_t>(n_ov), 0.0);
    e[static_cast<std::size_t>(idx)] = 1.0;
    V_sub.push_back(std::move(e));
  }

  GemmMode active_mode = opts.dynamic_precision ? opts.early_gemm_mode : opts.gemm_mode;
  bool promoted_to_final = !opts.dynamic_precision;
  std::size_t evaluated_count = 0;

  out.eigenvalues.assign(static_cast<std::size_t>(n_roots), 0.0);
  out.vectors.assign(static_cast<std::size_t>(n_roots),
                     std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
  out.residuals.assign(static_cast<std::size_t>(n_roots), 0.0);

  for (int iter = 1; iter <= opts.max_iter; ++iter) {
    while (evaluated_count < V_sub.size()) {
      std::vector<double> sig(static_cast<std::size_t>(n_ov), 0.0);
      matvec_fn(V_sub[evaluated_count].data(), sig.data(), active_mode);
      S_sub.push_back(std::move(sig));
      ++evaluated_count;
    }

    const int m = static_cast<int>(V_sub.size());
    std::vector<double> H_sub(static_cast<std::size_t>(m) * m, 0.0);
    for (int p = 0; p < m; ++p) {
      for (int q = p; q < m; ++q) {
        const double v =
            0.5 * (dot_vec(static_cast<std::size_t>(n_ov), V_sub[p].data(), S_sub[q].data()) +
                   dot_vec(static_cast<std::size_t>(n_ov), V_sub[q].data(), S_sub[p].data()));
        H_sub[static_cast<std::size_t>(p) * m + q] = v;
        H_sub[static_cast<std::size_t>(q) * m + p] = v;
      }
    }

    std::vector<double> evals(static_cast<std::size_t>(m), 0.0);
    std::vector<double> evecs(static_cast<std::size_t>(m) * m, 0.0);
    syevd(m, H_sub.data(), evals.data(), evecs.data(), opts.backend);

    const int k_check = std::min(n_roots, m);
    const int k_expand = std::min(n_track, m);
    double max_res = 0.0;
    std::vector<std::vector<double>> ritz_vecs(static_cast<std::size_t>(k_expand),
                                               std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
    std::vector<std::vector<double>> res_vecs(static_cast<std::size_t>(k_expand),
                                              std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
    std::vector<double> res_norms(static_cast<std::size_t>(k_expand), 0.0);

    for (int k = 0; k < k_expand; ++k) {
      const double theta = evals[static_cast<std::size_t>(k)];
      auto& xk = ritz_vecs[static_cast<std::size_t>(k)];
      auto& rk = res_vecs[static_cast<std::size_t>(k)];
      for (int p = 0; p < m; ++p) {
        const double cp = evecs[static_cast<std::size_t>(p) * m + k];
        for (int i = 0; i < n_ov; ++i) {
          xk[static_cast<std::size_t>(i)] += cp * V_sub[p][static_cast<std::size_t>(i)];
          rk[static_cast<std::size_t>(i)] +=
              cp * (S_sub[p][static_cast<std::size_t>(i)] - theta * V_sub[p][static_cast<std::size_t>(i)]);
        }
      }
      res_norms[static_cast<std::size_t>(k)] = norm2_vec(static_cast<std::size_t>(n_ov), rk.data());
      if (k < k_check) {
        max_res = std::max(max_res, res_norms[static_cast<std::size_t>(k)]);
        out.eigenvalues[static_cast<std::size_t>(k)] = theta;
        out.vectors[static_cast<std::size_t>(k)] = xk;
        out.residuals[static_cast<std::size_t>(k)] = res_norms[static_cast<std::size_t>(k)];
      }
    }

    out.iterations = iter;
    out.history.push_back(DavidsonIterationLog{iter, m, max_res, active_mode});

    if (!promoted_to_final && max_res < opts.fp64_switch_residual) {
      promoted_to_final = true;
      active_mode = opts.gemm_mode;
      // Rebuild subspace from current Ritz vectors and re-evaluate sigma in final precision!
      V_sub.clear();
      S_sub.clear();
      for (int k = 0; k < k_expand; ++k) {
        append_orthonormal_vector(n_ov, V_sub, ritz_vecs[static_cast<std::size_t>(k)], 1e-10);
      }
      evaluated_count = 0;
      continue;
    }

    if (k_check == n_roots && max_res <= opts.conv_tol) {
      out.converged = true;
      break;
    }
    if (m >= n_ov) {
      out.converged = (max_res <= std::max(opts.conv_tol * 10.0, 1e-10));
      break;
    }

    // Subspace collapse via exact linear combination if near capacity
    if (m + k_expand > max_sub) {
      std::vector<std::vector<double>> V_new;
      std::vector<std::vector<double>> S_new;
      for (int k = 0; k < k_expand; ++k) {
        std::vector<double> s_k(static_cast<std::size_t>(n_ov), 0.0);
        for (int p = 0; p < m; ++p) {
          const double cp = evecs[static_cast<std::size_t>(p) * m + k];
          for (int i = 0; i < n_ov; ++i) {
            s_k[static_cast<std::size_t>(i)] += cp * S_sub[p][static_cast<std::size_t>(i)];
          }
        }
        V_new.push_back(ritz_vecs[static_cast<std::size_t>(k)]);
        S_new.push_back(std::move(s_k));
      }
      V_sub = std::move(V_new);
      S_sub = std::move(S_new);
      evaluated_count = V_sub.size();
    }

    int added = 0;
    for (int k = 0; k < k_expand; ++k) {
      if (k < n_roots && res_norms[static_cast<std::size_t>(k)] <= 0.2 * opts.conv_tol) continue;
      if (res_norms[static_cast<std::size_t>(k)] <= 0.05 * opts.conv_tol) continue;
      const double theta = evals[static_cast<std::size_t>(k)];
      std::vector<double> corr(static_cast<std::size_t>(n_ov), 0.0);
      for (int i = 0; i < n_ov; ++i) {
        double denom = diag_approx[static_cast<std::size_t>(i)] - theta;
        if (std::abs(denom) < opts.denom_floor) {
          denom = (denom >= 0.0) ? opts.denom_floor : -opts.denom_floor;
        }
        corr[static_cast<std::size_t>(i)] =
            -res_vecs[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] / denom;
      }
      if (append_orthonormal_vector(n_ov, V_sub, std::move(corr))) ++added;
    }

    if (added == 0) {
      out.converged = (max_res <= std::max(opts.conv_tol * 10.0, 1e-9));
      break;
    }
  }

  return out;
}

// Stratmann-Scuseria-Frisch (1998) Paired Non-Hermitian Block-Davidson Solver for
// Hybrid TDDFT & TDHF:
//   (A + B)(X + Y) = \omega (X - Y),   (A - B)(X - Y) = \omega (X + Y),
// normalized to (X + Y)^T (X - Y) = X^T X - Y^T Y = 1.
struct PairedRpaDavidsonResult {
  bool converged = false;
  int iterations = 0;
  std::vector<double> omegas;                  // [n_states]
  std::vector<std::vector<double>> x_vectors;  // [n_states][n_ov]
  std::vector<std::vector<double>> y_vectors;  // [n_states][n_ov]
  std::vector<double> residuals;               // [n_states]
  std::vector<DavidsonIterationLog> history;
};

PairedRpaDavidsonResult run_ssf_rpa_davidson(OrbitalResponseOperator& op,
                                             const ResponseOptions& opts) {
  PairedRpaDavidsonResult out;
  const int n_ov = op.n_ov();
  const int n_roots = std::min(opts.n_states, n_ov);
  const int n_track = std::min(n_ov, std::max(n_roots + 3, n_roots));
  const int max_sub = std::max({opts.max_subspace, 10 * n_track, 32});
  const auto& delta_eps = op.delta_eps();

  const std::vector<int> guess_idx = select_initial_guess_indices(delta_eps, n_roots);
  std::vector<std::vector<double>> V_sub;
  std::vector<std::vector<double>> Sp_sub;
  std::vector<std::vector<double>> Sm_sub;
  V_sub.reserve(static_cast<std::size_t>(max_sub));
  Sp_sub.reserve(static_cast<std::size_t>(max_sub));
  Sm_sub.reserve(static_cast<std::size_t>(max_sub));

  for (int idx : guess_idx) {
    std::vector<double> e(static_cast<std::size_t>(n_ov), 0.0);
    e[static_cast<std::size_t>(idx)] = 1.0;
    V_sub.push_back(std::move(e));
  }

  GemmMode active_mode = opts.dynamic_precision ? opts.early_gemm_mode : opts.gemm_mode;
  bool promoted_to_final = !opts.dynamic_precision;
  std::size_t evaluated_count = 0;

  out.omegas.assign(static_cast<std::size_t>(n_roots), 0.0);
  out.x_vectors.assign(static_cast<std::size_t>(n_roots),
                       std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
  out.y_vectors.assign(static_cast<std::size_t>(n_roots),
                       std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
  out.residuals.assign(static_cast<std::size_t>(n_roots), 0.0);

  for (int iter = 1; iter <= opts.max_iter; ++iter) {
    while (evaluated_count < V_sub.size()) {
      std::vector<double> sp(static_cast<std::size_t>(n_ov), 0.0);
      std::vector<double> sm(static_cast<std::size_t>(n_ov), 0.0);
      op.apply_rpa_pair(V_sub[evaluated_count].data(), sp.data(), sm.data(), active_mode,
                        opts.ozaki_slices);
      Sp_sub.push_back(std::move(sp));
      Sm_sub.push_back(std::move(sm));
      ++evaluated_count;
    }

    const int m = static_cast<int>(V_sub.size());
    std::vector<double> Mp(static_cast<std::size_t>(m) * m, 0.0);
    std::vector<double> Mm(static_cast<std::size_t>(m) * m, 0.0);
    for (int p = 0; p < m; ++p) {
      for (int q = p; q < m; ++q) {
        const double vp =
            0.5 * (dot_vec(static_cast<std::size_t>(n_ov), V_sub[p].data(), Sp_sub[q].data()) +
                   dot_vec(static_cast<std::size_t>(n_ov), V_sub[q].data(), Sp_sub[p].data()));
        const double vm =
            0.5 * (dot_vec(static_cast<std::size_t>(n_ov), V_sub[p].data(), Sm_sub[q].data()) +
                   dot_vec(static_cast<std::size_t>(n_ov), V_sub[q].data(), Sm_sub[p].data()));
        Mp[static_cast<std::size_t>(p) * m + q] = vp;
        Mp[static_cast<std::size_t>(q) * m + p] = vp;
        Mm[static_cast<std::size_t>(p) * m + q] = vm;
        Mm[static_cast<std::size_t>(q) * m + p] = vm;
      }
    }

    // Solve reduced paired eigenproblem M^+ u = \omega v, M^- v = \omega u via
    // exact Cholesky/spectral square root of M^-: W = (M^-)^{1/2} M^+ (M^-)^{1/2} f = \omega^2 f.
    std::vector<double> dm_vals(static_cast<std::size_t>(m), 0.0);
    std::vector<double> Qm(static_cast<std::size_t>(m) * m, 0.0);
    syevd(m, Mm.data(), dm_vals.data(), Qm.data(), opts.backend);

    for (double val : dm_vals) {
      if (!(val > 1e-10)) {
        throw std::runtime_error(
            "LinearResponseDriver (TDDFT/TDHF): ground state has singlet/RPA instability "
            "(A - B is not positive definite)");
      }
    }

    std::vector<double> Mm_sqrt(static_cast<std::size_t>(m) * m, 0.0);
    std::vector<double> Mm_isqrt(static_cast<std::size_t>(m) * m, 0.0);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        double s_pos = 0.0, s_inv = 0.0;
        for (int r = 0; r < m; ++r) {
          const double qir = Qm[static_cast<std::size_t>(i) * m + r];
          const double qjr = Qm[static_cast<std::size_t>(j) * m + r];
          const double sq = std::sqrt(dm_vals[static_cast<std::size_t>(r)]);
          s_pos += qir * sq * qjr;
          s_inv += qir * (1.0 / sq) * qjr;
        }
        Mm_sqrt[static_cast<std::size_t>(i) * m + j] = s_pos;
        Mm_isqrt[static_cast<std::size_t>(i) * m + j] = s_inv;
      }
    }

    std::vector<double> tmp_mm(static_cast<std::size_t>(m) * m, 0.0);
    std::vector<double> W_sub(static_cast<std::size_t>(m) * m, 0.0);
    gemm(m, m, m, 1.0, Mm_sqrt.data(), false, Mp.data(), false, 0.0, tmp_mm.data(), Backend::Host);
    gemm(m, m, m, 1.0, tmp_mm.data(), false, Mm_sqrt.data(), false, 0.0, W_sub.data(),
         Backend::Host);
    for (int i = 0; i < m; ++i) {
      for (int j = i + 1; j < m; ++j) {
        const double avg =
            0.5 * (W_sub[static_cast<std::size_t>(i) * m + j] +
                   W_sub[static_cast<std::size_t>(j) * m + i]);
        W_sub[static_cast<std::size_t>(i) * m + j] = avg;
        W_sub[static_cast<std::size_t>(j) * m + i] = avg;
      }
    }

    std::vector<double> omega2(static_cast<std::size_t>(m), 0.0);
    std::vector<double> F_sub(static_cast<std::size_t>(m) * m, 0.0);
    syevd(m, W_sub.data(), omega2.data(), F_sub.data(), opts.backend);

    const int k_check = std::min(n_roots, m);
    const int k_expand = std::min(n_track, m);
    double max_res = 0.0;

    std::vector<std::vector<double>> U_full(static_cast<std::size_t>(k_expand),
                                            std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
    std::vector<std::vector<double>> V_full(static_cast<std::size_t>(k_expand),
                                            std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
    std::vector<std::vector<double>> rX_full(static_cast<std::size_t>(k_expand),
                                             std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
    std::vector<std::vector<double>> rY_full(static_cast<std::size_t>(k_expand),
                                             std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
    std::vector<double> omegas_k(static_cast<std::size_t>(k_expand), 0.0);
    std::vector<double> res_norms(static_cast<std::size_t>(k_expand), 0.0);

    for (int k = 0; k < k_expand; ++k) {
      if (!(omega2[static_cast<std::size_t>(k)] > 1e-12)) {
        throw std::runtime_error("LinearResponseDriver (TDDFT/TDHF): non-positive omega^2 encountered");
      }
      const double w = std::sqrt(omega2[static_cast<std::size_t>(k)]);
      omegas_k[static_cast<std::size_t>(k)] = w;
      const double inv_sw = 1.0 / std::sqrt(w);
      const double sw = std::sqrt(w);

      std::vector<double> uk_sub(static_cast<std::size_t>(m), 0.0);
      std::vector<double> vk_sub(static_cast<std::size_t>(m), 0.0);
      for (int p = 0; p < m; ++p) {
        double su = 0.0, sv = 0.0;
        for (int r = 0; r < m; ++r) {
          const double fr = F_sub[static_cast<std::size_t>(r) * m + k];
          su += Mm_sqrt[static_cast<std::size_t>(p) * m + r] * fr;
          sv += Mm_isqrt[static_cast<std::size_t>(p) * m + r] * fr;
        }
        uk_sub[static_cast<std::size_t>(p)] = inv_sw * su;
        vk_sub[static_cast<std::size_t>(p)] = sw * sv;
      }

      auto& Uk = U_full[static_cast<std::size_t>(k)];
      auto& Vk = V_full[static_cast<std::size_t>(k)];
      auto& rX = rX_full[static_cast<std::size_t>(k)];
      auto& rY = rY_full[static_cast<std::size_t>(k)];

      for (int p = 0; p < m; ++p) {
        const double up = uk_sub[static_cast<std::size_t>(p)];
        const double vp = vk_sub[static_cast<std::size_t>(p)];
        for (int i = 0; i < n_ov; ++i) {
          const std::size_t idx = static_cast<std::size_t>(i);
          Uk[idx] += up * V_sub[p][idx];
          Vk[idx] += vp * V_sub[p][idx];
        }
      }

      // r^+ = \Sigma^+ u - \omega V_full,   r^- = \Sigma^- v - \omega U_full
      // r^X = 0.5 (r^+ + r^-),              r^Y = 0.5 (r^+ - r^-)
      for (int p = 0; p < m; ++p) {
        const double up = uk_sub[static_cast<std::size_t>(p)];
        const double vp = vk_sub[static_cast<std::size_t>(p)];
        for (int i = 0; i < n_ov; ++i) {
          const std::size_t idx = static_cast<std::size_t>(i);
          const double rp = up * Sp_sub[p][idx];
          const double rm = vp * Sm_sub[p][idx];
          rX[idx] += 0.5 * (rp + rm);
          rY[idx] += 0.5 * (rp - rm);
        }
      }
      for (int i = 0; i < n_ov; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        rX[idx] -= 0.5 * w * (Vk[idx] + Uk[idx]);
        rY[idx] -= 0.5 * w * (Vk[idx] - Uk[idx]);
      }

      const double nx = norm2_vec(static_cast<std::size_t>(n_ov), rX.data());
      const double ny = norm2_vec(static_cast<std::size_t>(n_ov), rY.data());
      const double rtot = std::hypot(nx, ny);
      res_norms[static_cast<std::size_t>(k)] = rtot;

      if (k < k_check) {
        max_res = std::max(max_res, rtot);
        out.omegas[static_cast<std::size_t>(k)] = w;
        out.residuals[static_cast<std::size_t>(k)] = rtot;
        for (int i = 0; i < n_ov; ++i) {
          const std::size_t idx = static_cast<std::size_t>(i);
          out.x_vectors[static_cast<std::size_t>(k)][idx] = 0.5 * (Uk[idx] + Vk[idx]);
          out.y_vectors[static_cast<std::size_t>(k)][idx] = 0.5 * (Uk[idx] - Vk[idx]);
        }
      }
    }

    out.iterations = iter;
    out.history.push_back(DavidsonIterationLog{iter, m, max_res, active_mode});

    if (!promoted_to_final && max_res < opts.fp64_switch_residual) {
      promoted_to_final = true;
      active_mode = opts.gemm_mode;
      V_sub.clear();
      Sp_sub.clear();
      Sm_sub.clear();
      for (int k = 0; k < k_expand; ++k) {
        append_orthonormal_vector(n_ov, V_sub, U_full[static_cast<std::size_t>(k)], 1e-9);
        append_orthonormal_vector(n_ov, V_sub, V_full[static_cast<std::size_t>(k)], 1e-9);
      }
      evaluated_count = 0;
      continue;
    }

    if (k_check == n_roots && max_res <= opts.conv_tol) {
      out.converged = true;
      break;
    }
    if (m >= n_ov) {
      out.converged = (max_res <= std::max(opts.conv_tol * 10.0, 1e-10));
      break;
    }

    // Subspace restart preserving span{U_1..U_p, V_1..V_p} with zero extra JK/Fxc calls!
    if (m + 2 * k_expand > max_sub) {
      // Orthonormalize coefficient columns in R^m, then transform V_sub, Sp_sub, Sm_sub!
      std::vector<std::vector<double>> R_cols;
      auto add_sub_col = [&](std::vector<double> col) {
        double n0 = norm2_vec(static_cast<std::size_t>(m), col.data());
        if (!(n0 > 1e-12)) return;
        for (double& x : col) x /= n0;
        for (int pass = 0; pass < 2; ++pass) {
          for (const auto& rc : R_cols) {
            const double pr = dot_vec(static_cast<std::size_t>(m), rc.data(), col.data());
            for (int p = 0; p < m; ++p) col[static_cast<std::size_t>(p)] -= pr * rc[static_cast<std::size_t>(p)];
          }
          const double n1 = norm2_vec(static_cast<std::size_t>(m), col.data());
          if (!(n1 > 1e-8)) return;
          for (double& x : col) x /= n1;
        }
        R_cols.push_back(std::move(col));
      };

      for (int k = 0; k < k_expand; ++k) {
        std::vector<double> cu(static_cast<std::size_t>(m), 0.0);
        std::vector<double> cv(static_cast<std::size_t>(m), 0.0);
        for (int p = 0; p < m; ++p) {
          for (int r = 0; r < m; ++r) {
            const double fr = F_sub[static_cast<std::size_t>(r) * m + k];
            cu[static_cast<std::size_t>(p)] += Mm_sqrt[static_cast<std::size_t>(p) * m + r] * fr;
            cv[static_cast<std::size_t>(p)] += Mm_isqrt[static_cast<std::size_t>(p) * m + r] * fr;
          }
        }
        add_sub_col(std::move(cu));
        add_sub_col(std::move(cv));
      }

      std::vector<std::vector<double>> V_new(R_cols.size(), std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
      std::vector<std::vector<double>> Sp_new(R_cols.size(), std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
      std::vector<std::vector<double>> Sm_new(R_cols.size(), std::vector<double>(static_cast<std::size_t>(n_ov), 0.0));
      for (std::size_t c = 0; c < R_cols.size(); ++c) {
        for (int p = 0; p < m; ++p) {
          const double coef = R_cols[c][static_cast<std::size_t>(p)];
          for (int i = 0; i < n_ov; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            V_new[c][idx] += coef * V_sub[p][idx];
            Sp_new[c][idx] += coef * Sp_sub[p][idx];
            Sm_new[c][idx] += coef * Sm_sub[p][idx];
          }
        }
      }
      V_sub = std::move(V_new);
      Sp_sub = std::move(Sp_new);
      Sm_sub = std::move(Sm_new);
      evaluated_count = V_sub.size();
    }

    int added = 0;
    for (int k = 0; k < k_expand; ++k) {
      if (k < n_roots && res_norms[static_cast<std::size_t>(k)] <= 0.2 * opts.conv_tol) continue;
      if (res_norms[static_cast<std::size_t>(k)] <= 0.05 * opts.conv_tol) continue;

      const double w = omegas_k[static_cast<std::size_t>(k)];
      std::vector<double> dU(static_cast<std::size_t>(n_ov), 0.0);
      std::vector<double> dV(static_cast<std::size_t>(n_ov), 0.0);
      for (int i = 0; i < n_ov; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        double d_minus = delta_eps[idx] - w;
        if (std::abs(d_minus) < opts.denom_floor) {
          d_minus = (d_minus >= 0.0) ? opts.denom_floor : -opts.denom_floor;
        }
        const double d_plus = std::max(delta_eps[idx] + w, opts.denom_floor);
        const double dx = -rX_full[static_cast<std::size_t>(k)][idx] / d_minus;
        const double dy = -rY_full[static_cast<std::size_t>(k)][idx] / d_plus;
        dU[idx] = dx + dy;
        dV[idx] = dx - dy;
      }
      if (append_orthonormal_vector(n_ov, V_sub, std::move(dU))) ++added;
      if (append_orthonormal_vector(n_ov, V_sub, std::move(dV))) ++added;
    }

    if (added == 0) {
      out.converged = (max_res <= std::max(opts.conv_tol * 10.0, 1e-9));
      break;
    }
  }

  return out;
}

}  // namespace

// ===========================================================================
// 1. Public Dipole Integrals Implementation
// ===========================================================================

std::array<std::vector<double>, 3> dipole_matrices(const Basis& basis,
                                                   std::array<double, 3> origin,
                                                   Backend backend) {
  if (!backend_available(backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(backend) + "' is not available");
  }
  const HostDipoleBasisTable table = pack_dipole_basis(basis);
  const std::size_t n2 = static_cast<std::size_t>(basis.nao()) * basis.nao();
  std::array<std::vector<double>, 3> out = {
      std::vector<double>(n2, 0.0),
      std::vector<double>(n2, 0.0),
      std::vector<double>(n2, 0.0),
  };
  if (n2 == 0) return out;

  if (backend == Backend::Host) {
    const detail::BasisDeviceView v = table.view();
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<std::size_t>(v.nshell) * (v.nshell + 1) / 2);
    for (int i = 0; i < v.nshell; ++i) {
      for (int j = 0; j <= i; ++j) pairs.emplace_back(i, j);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 16)
#endif
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(pairs.size()); ++idx) {
      const auto [ish, jsh] = pairs[static_cast<std::size_t>(idx)];
      detail::eval_dipole_pair(v, ish, jsh, origin[0], origin[1], origin[2], out[0].data(),
                               out[1].data(), out[2].data());
    }
    return out;
  }

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::run_dipole_matrices_cuda(table.view(), origin, out);
    return out;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  throw std::runtime_error("gpu4dft was built without HIP");
}

std::array<double, 3> molecular_dipole_moment(const Basis& basis, std::span<const Atom> atoms,
                                              const double* dm_host,
                                              std::array<double, 3> origin, Backend backend) {
  const auto ao_dip = dipole_matrices(basis, origin, backend);
  const std::size_t n2 = static_cast<std::size_t>(basis.nao()) * basis.nao();
  std::array<double, 3> dip = {0.0, 0.0, 0.0};
  for (const Atom& a : atoms) {
    for (int c = 0; c < 3; ++c) {
      dip[c] += static_cast<double>(a.atomic_number) * (a.center[c] - origin[c]);
    }
  }
  if (dm_host != nullptr && n2 > 0) {
    for (int c = 0; c < 3; ++c) {
      dip[c] += dot_vec(n2, dm_host, ao_dip[c].data());
    }
  }
  return dip;
}

// ===========================================================================
// 2. LinearResponseDriver & Callback Factories Implementation
// ===========================================================================

struct LinearResponseDriver::Impl {
  ResponseOptions opts;
  DeviceMemoryPool* ext_pool = nullptr;

  Impl(ResponseOptions o, DeviceMemoryPool* p) : opts(std::move(o)), ext_pool(p) {}
};

LinearResponseDriver::LinearResponseDriver(ResponseOptions opts, DeviceMemoryPool* pool)
    : impl_(std::make_unique<Impl>(std::move(opts), pool)) {}

LinearResponseDriver::~LinearResponseDriver() = default;
LinearResponseDriver::LinearResponseDriver(LinearResponseDriver&&) noexcept = default;
LinearResponseDriver& LinearResponseDriver::operator=(LinearResponseDriver&&) noexcept = default;

const ResponseOptions& LinearResponseDriver::options() const noexcept { return impl_->opts; }
void LinearResponseDriver::set_options(ResponseOptions opts) { impl_->opts = std::move(opts); }

ResponseResult LinearResponseDriver::run(
    int nao, int nmo, int n_occ, const double* mo_energies_host, const double* mo_coeff_host,
    const ResponseCallbacks& callbacks,
    const std::array<std::vector<double>, 3>* ao_dipoles_host) {
  const auto t0 = SteadyClock::now();
  const ResponseOptions& opts = impl_->opts;

  if (!backend_available(opts.backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(opts.backend) + "' is not available");
  }
  if (nao <= 0 || nmo <= 0 || n_occ <= 0 || n_occ >= nmo) {
    throw std::invalid_argument("LinearResponseDriver::run: invalid (nao, nmo, n_occ)");
  }
  if (mo_energies_host == nullptr || mo_coeff_host == nullptr) {
    throw std::invalid_argument("LinearResponseDriver::run: null MO energies or coefficients");
  }
  if (opts.n_states <= 0) {
    throw std::invalid_argument("LinearResponseDriver::run: n_states must be positive");
  }

  OrbitalResponseOperator op(nao, nmo, n_occ, mo_energies_host, mo_coeff_host,
                             opts.exact_exchange_fraction, opts.backend, callbacks,
                             impl_->ext_pool);

  const int n_vir = op.n_vir();
  const int n_ov = op.n_ov();
  const int n_roots = std::min(opts.n_states, n_ov);

  ResponseResult res;
  res.nao = nao;
  res.nmo = nmo;
  res.n_occ = n_occ;
  res.n_vir = n_vir;
  res.n_states = n_roots;
  res.excitation_energies.assign(static_cast<std::size_t>(n_roots), 0.0);
  res.excitation_energies_ev.assign(static_cast<std::size_t>(n_roots), 0.0);
  res.oscillator_strengths.assign(static_cast<std::size_t>(n_roots), 0.0);
  res.transition_dipoles.assign(static_cast<std::size_t>(n_roots), {0.0, 0.0, 0.0});
  res.x_vectors.assign(static_cast<std::size_t>(n_roots) * n_ov, 0.0);
  res.y_vectors.assign(static_cast<std::size_t>(n_roots) * n_ov, 0.0);
  res.states.resize(static_cast<std::size_t>(n_roots));

  std::vector<double> state_residuals(static_cast<std::size_t>(n_roots), 0.0);

  if (opts.method == ResponseMethod::TDA) {
    auto hd = run_hermitian_davidson(
        n_ov, n_roots, op.delta_eps(), opts,
        [&](const double* x, double* y, GemmMode m) { op.apply_tda(x, y, m, opts.ozaki_slices); });
    res.converged = hd.converged;
    res.iterations = hd.iterations;
    res.history = std::move(hd.history);
    for (int k = 0; k < n_roots; ++k) {
      res.excitation_energies[static_cast<std::size_t>(k)] = hd.eigenvalues[static_cast<std::size_t>(k)];
      state_residuals[static_cast<std::size_t>(k)] = hd.residuals[static_cast<std::size_t>(k)];
      std::copy(hd.vectors[static_cast<std::size_t>(k)].begin(),
                hd.vectors[static_cast<std::size_t>(k)].end(),
                res.x_vectors.begin() + static_cast<std::ptrdiff_t>(k) * n_ov);
    }
  } else if (std::abs(opts.exact_exchange_fraction) < 1e-14) {
    // Pure DFT Casida Hermitianized eigenproblem: \Omega F = \omega^2 F
    std::vector<double> omega_diag(static_cast<std::size_t>(n_ov));
    for (int i = 0; i < n_ov; ++i) {
      const double de = op.delta_eps()[static_cast<std::size_t>(i)];
      omega_diag[static_cast<std::size_t>(i)] = de * de;
    }
    ResponseOptions cas_opts = opts;
    // Note: residual of \Omega F = \omega^2 F scales as ~2*\omega * r_RPA, so tighten slightly
    cas_opts.conv_tol = std::min(opts.conv_tol, 1e-8);
    auto hd = run_hermitian_davidson(
        n_ov, n_roots, omega_diag, cas_opts, [&](const double* v, double* y, GemmMode m) {
          op.apply_casida_omega(v, y, m, opts.ozaki_slices);
        });
    res.converged = hd.converged;
    res.iterations = hd.iterations;
    res.history = std::move(hd.history);
    const auto& sq_eps = op.sqrt_eps();
    const auto& isq_eps = op.inv_sqrt_eps();
    for (int k = 0; k < n_roots; ++k) {
      const double w2 = hd.eigenvalues[static_cast<std::size_t>(k)];
      if (!(w2 > 0.0)) {
        throw std::runtime_error("LinearResponseDriver (Casida TDDFT): non-positive omega^2");
      }
      const double w = std::sqrt(w2);
      const double inv_sw = 1.0 / std::sqrt(w);
      const double sw = std::sqrt(w);
      res.excitation_energies[static_cast<std::size_t>(k)] = w;
      state_residuals[static_cast<std::size_t>(k)] = hd.residuals[static_cast<std::size_t>(k)];
      const auto& Fk = hd.vectors[static_cast<std::size_t>(k)];
      double* Xk = res.x_vectors.data() + static_cast<std::size_t>(k) * n_ov;
      double* Yk = res.y_vectors.data() + static_cast<std::size_t>(k) * n_ov;
      for (int i = 0; i < n_ov; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        const double uk = inv_sw * sq_eps[idx] * Fk[idx];
        const double vk = sw * isq_eps[idx] * Fk[idx];
        Xk[idx] = 0.5 * (uk + vk);
        Yk[idx] = 0.5 * (uk - vk);
      }
    }
  } else {
    // Hybrid TDDFT / TDHF Paired SSF Block-Davidson
    auto rd = run_ssf_rpa_davidson(op, opts);
    res.converged = rd.converged;
    res.iterations = rd.iterations;
    res.history = std::move(rd.history);
    for (int k = 0; k < n_roots; ++k) {
      res.excitation_energies[static_cast<std::size_t>(k)] = rd.omegas[static_cast<std::size_t>(k)];
      state_residuals[static_cast<std::size_t>(k)] = rd.residuals[static_cast<std::size_t>(k)];
      std::copy(rd.x_vectors[static_cast<std::size_t>(k)].begin(),
                rd.x_vectors[static_cast<std::size_t>(k)].end(),
                res.x_vectors.begin() + static_cast<std::ptrdiff_t>(k) * n_ov);
      std::copy(rd.y_vectors[static_cast<std::size_t>(k)].begin(),
                rd.y_vectors[static_cast<std::size_t>(k)].end(),
                res.y_vectors.begin() + static_cast<std::ptrdiff_t>(k) * n_ov);
    }
  }

  res.matvec_count = op.matvec_count();

  // Pre-project AO electric dipoles into MO occ-vir basis: mu_mo^{(c)} = C_o^T \mu^{(c)} C_v [n_occ x n_vir]
  std::array<std::vector<double>, 3> mu_mo;
  const bool have_dipoles =
      (ao_dipoles_host != nullptr) &&
      ((*ao_dipoles_host)[0].size() == static_cast<std::size_t>(nao) * nao);
  if (have_dipoles) {
    std::vector<double> tmp_uv(static_cast<std::size_t>(nao) * n_vir, 0.0);
    for (int c = 0; c < 3; ++c) {
      mu_mo[c].assign(static_cast<std::size_t>(n_ov), 0.0);
      gemm(nao, n_vir, nao, 1.0, (*ao_dipoles_host)[c].data(), false, op.Cv_host().data(), false,
           0.0, tmp_uv.data(), opts.backend, GemmMode::FP64);
      gemm(n_occ, n_vir, nao, 1.0, op.Co_host().data(), true, tmp_uv.data(), false, 0.0,
           mu_mo[c].data(), opts.backend, GemmMode::FP64);
    }
  }

  const double sqrt2 = std::sqrt(2.0);
  for (int k = 0; k < n_roots; ++k) {
    const double w = res.excitation_energies[static_cast<std::size_t>(k)];
    const double* Xk = res.x_vectors.data() + static_cast<std::size_t>(k) * n_ov;
    const double* Yk = res.y_vectors.data() + static_cast<std::size_t>(k) * n_ov;

    // Fix deterministic global phase: largest-magnitude X_{ia} component positive
    int dom_ia = 0;
    double max_abs_x = -1.0;
    double dom_wt = 0.0;
    for (int ia = 0; ia < n_ov; ++ia) {
      const double ax = std::abs(Xk[ia]);
      if (ax > max_abs_x) {
        max_abs_x = ax;
        dom_ia = ia;
        dom_wt = Xk[ia] * Xk[ia] - Yk[ia] * Yk[ia];
      }
    }
    if (Xk[dom_ia] < 0.0) {
      double* Xk_mut = res.x_vectors.data() + static_cast<std::size_t>(k) * n_ov;
      double* Yk_mut = res.y_vectors.data() + static_cast<std::size_t>(k) * n_ov;
      for (int ia = 0; ia < n_ov; ++ia) {
        Xk_mut[ia] = -Xk_mut[ia];
        Yk_mut[ia] = -Yk_mut[ia];
      }
    }

    std::array<double, 3> tdip = {0.0, 0.0, 0.0};
    double f_osc = 0.0;
    if (have_dipoles) {
      for (int c = 0; c < 3; ++c) {
        double s = 0.0;
        for (int ia = 0; ia < n_ov; ++ia) {
          s += mu_mo[c][static_cast<std::size_t>(ia)] * (Xk[ia] + Yk[ia]);
        }
        tdip[c] = sqrt2 * s;
      }
      const double d2 = tdip[0] * tdip[0] + tdip[1] * tdip[1] + tdip[2] * tdip[2];
      f_osc = (2.0 / 3.0) * w * d2;
    }

    res.excitation_energies_ev[static_cast<std::size_t>(k)] = w * kHartreeToEv;
    res.transition_dipoles[static_cast<std::size_t>(k)] = tdip;
    res.oscillator_strengths[static_cast<std::size_t>(k)] = f_osc;

    ExcitedState st;
    st.state_index = k;
    st.energy_hartree = w;
    st.energy_ev = w * kHartreeToEv;
    st.transition_dipole = tdip;
    st.oscillator_strength = f_osc;
    st.residual_norm = state_residuals[static_cast<std::size_t>(k)];
    st.dominant_occ = dom_ia / n_vir;
    st.dominant_vir = n_occ + (dom_ia % n_vir);
    st.dominant_weight = dom_wt;
    res.states[static_cast<std::size_t>(k)] = st;
  }

  res.wall_time_s = elapsed_sec(t0, SteadyClock::now());
  return res;
}

ResponseCallbacks make_in_core_eri_response_callbacks(int nao, const double* h_eri, Backend backend,
                                                      bool need_k, GemmMode gemm_mode,
                                                      int ozaki_slices) {
  if (nao <= 0 || h_eri == nullptr) {
    throw std::invalid_argument("make_in_core_eri_response_callbacks: invalid nao or null h_eri");
  }

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (!backend_available(Backend::Cuda)) {
      throw std::runtime_error("make_in_core_eri_response_callbacks: CUDA backend unavailable");
    }
    auto gpu_state = detail::create_cuda_incore_response_jk(nao, h_eri, need_k);
    ResponseCallbacks cb;
    cb.device_resident = true;
    cb.build_jk = [gpu_state, gemm_mode, ozaki_slices](const double* d_dm, double* d_vj,
                                                       double* d_vk, void* stream) {
      detail::eval_cuda_incore_response_jk(*gpu_state, d_dm, d_vj, d_vk, gemm_mode, ozaki_slices,
                                           reinterpret_cast<StreamHandle>(stream));
    };
    return cb;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const std::size_t n4 = nao2 * nao2;
  auto eri_j = std::make_shared<std::vector<double>>(h_eri, h_eri + n4);
  std::shared_ptr<std::vector<double>> eri_k;
  if (need_k) {
    eri_k = std::make_shared<std::vector<double>>(n4, 0.0);
    for (int u = 0; u < nao; ++u) {
      for (int v = 0; v < nao; ++v) {
        const std::size_t uv = static_cast<std::size_t>(u) * nao + v;
        for (int l = 0; l < nao; ++l) {
          for (int s = 0; s < nao; ++s) {
            const std::size_t ls = static_cast<std::size_t>(l) * nao + s;
            const std::size_t src =
                ((static_cast<std::size_t>(u) * nao + l) * nao + v) * nao + s;
            (*eri_k)[uv * nao2 + ls] = (*eri_j)[src];
          }
        }
      }
    }
  }

  ResponseCallbacks cb;
  cb.device_resident = false;
  cb.build_jk = [nao2, eri_j, eri_k, gemm_mode, ozaki_slices](const double* dm, double* vj,
                                                              double* vk, void*) {
    const int n2 = static_cast<int>(nao2);
    if (vj != nullptr) {
      gemm(n2, 1, n2, 1.0, eri_j->data(), false, dm, false, 0.0, vj, Backend::Host, gemm_mode,
           ozaki_slices);
    }
    if (vk != nullptr && eri_k) {
      gemm(n2, 1, n2, 1.0, eri_k->data(), false, dm, false, 0.0, vk, Backend::Host, gemm_mode,
           ozaki_slices);
    }
  };
  return cb;
}

ResponseCallbacks make_direct_jk_response_callbacks(const Basis& basis, Backend backend,
                                                    bool need_k, double schwarz_cutoff,
                                                    bool force_on_the_fly_host) {
  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (!backend_available(Backend::Cuda)) {
      throw std::runtime_error("make_direct_jk_response_callbacks: CUDA backend unavailable");
    }
    auto gpu_jk = detail::create_cuda_direct_response_jk(basis, need_k, schwarz_cutoff);
    ResponseCallbacks cb;
    cb.device_resident = true;
    cb.build_jk = [gpu_jk](const double* d_dm, double* d_vj, double* d_vk, void* stream) {
      detail::eval_cuda_direct_response_jk(*gpu_jk, d_dm, d_vj, d_vk,
                                           reinterpret_cast<StreamHandle>(stream));
    };
    return cb;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  // On Host, for moderate basis sizes (nao <= 56, <= 78 MB ERI tensor), precompute GPU4DFT's
  // exact spherical ERI tensor once and contract via cache-blocked OpenMP GEMM unless on-the-fly
  // Rys quartet evaluation is explicitly requested.
  if (!force_on_the_fly_host && basis.nao() > 0 && basis.nao() <= 56) {
    const std::vector<double> h_eri = eri_tensor(basis, Backend::Host, Precision::FP64);
    return make_in_core_eri_response_callbacks(basis.nao(), h_eri.data(), Backend::Host, need_k);
  }

  auto host_jk = std::make_shared<HostDirectResponseJK>(basis, schwarz_cutoff);
  ResponseCallbacks cb;
  cb.device_resident = false;
  cb.build_jk = [host_jk](const double* dm, double* vj, double* vk, void*) {
    host_jk->eval(dm, vj, vk);
  };
  return cb;
}

void attach_fxc_response_callback(ResponseCallbacks& callbacks, const Basis& basis,
                                  const MolecularGrid& grid, std::string_view xc_name,
                                  const double* dm0_host, Backend backend, double rho_cutoff) {
  if (is_hf_xc_name(xc_name)) {
    callbacks.build_fxc = nullptr;
    callbacks.fxc_device_resident = false;
    return;
  }
  if (dm0_host == nullptr) {
    throw std::invalid_argument("attach_fxc_response_callback: dm0_host is null");
  }

  const auto fid = xc_gen::parse_xc_functional(xc_name);
  const int func_id = static_cast<int>(fid);
  const int family = static_cast<int>(xc_gen::xc_functional_family(fid));
  const detail::FlatBasisData flat(basis);

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (!backend_available(Backend::Cuda)) {
      throw std::runtime_error("attach_fxc_response_callback: CUDA backend unavailable");
    }
    auto gpu_fxc =
        detail::create_cuda_fxc_engine(flat, grid, func_id, family, dm0_host, rho_cutoff);
    callbacks.fxc_device_resident = true;
    callbacks.build_fxc = [gpu_fxc](const double* d_sym_dm, double* d_fxc_out, void* stream) {
      detail::eval_cuda_fxc_device(*gpu_fxc, d_sym_dm, d_fxc_out,
                                   reinterpret_cast<StreamHandle>(stream));
    };
    return;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  auto host_fxc = std::make_shared<HostFxcEngine>(flat, grid, func_id, family, dm0_host, rho_cutoff);
  callbacks.fxc_device_resident = false;
  callbacks.build_fxc = [host_fxc](const double* sym_dm, double* fxc_out, void*) {
    host_fxc->eval(sym_dm, fxc_out);
  };
}

ResponseResult run_linear_response(const Basis& basis, std::span<const Atom> atoms, int n_occ,
                                   const ScfResult& scf_res, std::string_view xc_name,
                                   const ResponseOptions& opts,
                                   std::span<const double> external_grid_coords,
                                   std::span<const double> external_grid_weights,
                                   const GridOptions& grid_opts) {
  if (scf_res.mo_energies.empty() || scf_res.mo_coefficients.empty()) {
    throw std::invalid_argument(
        "run_linear_response requires canonical MO energies and coefficients (Diag SCF result)");
  }

  ResponseOptions eff_opts = opts;
  const bool is_hf = is_hf_xc_name(xc_name);
  eff_opts.exact_exchange_fraction = is_hf ? 1.0 : xc_exact_exchange_fraction(xc_name);

  const bool need_k = (std::abs(eff_opts.exact_exchange_fraction) > 0.0);
  ResponseCallbacks cb =
      make_direct_jk_response_callbacks(basis, eff_opts.backend, need_k, 1e-13);

  if (!is_hf) {
    MolecularGrid grid;
    if (!external_grid_coords.empty() && !external_grid_weights.empty()) {
      grid = MolecularGrid::from_points(external_grid_coords, external_grid_weights, basis,
                                        grid_opts.max_batch_points, grid_opts.ao_cutoff);
    } else {
      std::vector<XcAtom> xc_atoms;
      xc_atoms.reserve(atoms.size());
      for (const Atom& a : atoms) xc_atoms.push_back(XcAtom{a.atomic_number, a.center});
      GridOptions gopts = grid_opts;
      gopts.backend = eff_opts.backend;
      grid = MolecularGrid(xc_atoms, basis, gopts);
    }
    attach_fxc_response_callback(cb, basis, grid, xc_name, scf_res.density_matrix.data(),
                                 eff_opts.backend);
  }

  const auto ao_dip = dipole_matrices(basis, {0.0, 0.0, 0.0}, eff_opts.backend);
  LinearResponseDriver drv(eff_opts);
  return drv.run(scf_res.nao, scf_res.nmo, n_occ, scf_res.mo_energies.data(),
                 scf_res.mo_coefficients.data(), cb, &ao_dip);
}

// ===========================================================================
// 3. CPHF / Z-Vector & Static Dipole Polarizability Implementation
// ===========================================================================

CphfResult solve_cphf(int nao, int nmo, int n_occ, const double* mo_energies_host,
                      const double* mo_coeff_host, const double* rhs_mo_host,
                      const ResponseCallbacks& callbacks, const CphfOptions& opts) {
  if (nao <= 0 || nmo <= 0 || n_occ <= 0 || n_occ >= nmo || rhs_mo_host == nullptr) {
    throw std::invalid_argument("solve_cphf: invalid dimensions or null rhs_mo_host");
  }
  OrbitalResponseOperator op(nao, nmo, n_occ, mo_energies_host, mo_coeff_host,
                             opts.exact_exchange_fraction, opts.backend, callbacks);
  const int n_vir = op.n_vir();
  const int n_ov = op.n_ov();
  const std::size_t n = static_cast<std::size_t>(n_ov);
  const auto& de = op.delta_eps();

  auto apply_mat = [&](const double* x, double* y) {
    if (opts.symmetric_perturbation) {
      op.apply_rpa_pair(x, y, nullptr, opts.gemm_mode, opts.ozaki_slices);
    } else {
      op.apply_rpa_pair(x, nullptr, y, opts.gemm_mode, opts.ozaki_slices);
    }
  };

  // Preconditioned Conjugate Gradient with diagonal preconditioner M^{-1}_{ia} = 1 / \Delta\eps_{ia}
  CphfResult res;
  res.z_mo.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    res.z_mo[i] = rhs_mo_host[i] / std::max(de[i], 1e-8);
  }

  std::vector<double> ax(n, 0.0), r(n, 0.0), z(n, 0.0), p(n, 0.0), ap(n, 0.0);
  apply_mat(res.z_mo.data(), ax.data());
  for (std::size_t i = 0; i < n; ++i) {
    r[i] = rhs_mo_host[i] - ax[i];
    z[i] = r[i] / std::max(de[i], 1e-8);
    p[i] = z[i];
  }
  double rz_old = dot_vec(n, r.data(), z.data());
  res.residual_norm = norm2_vec(n, r.data());
  if (res.residual_norm <= opts.conv_tol) {
    res.converged = true;
  } else {
    for (int iter = 1; iter <= opts.max_iter; ++iter) {
      apply_mat(p.data(), ap.data());
      const double pap = dot_vec(n, p.data(), ap.data());
      if (std::abs(pap) < 1e-30) break;
      const double alpha = rz_old / pap;
      for (std::size_t i = 0; i < n; ++i) {
        res.z_mo[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
      }
      res.iterations = iter;
      res.residual_norm = norm2_vec(n, r.data());
      if (res.residual_norm <= opts.conv_tol) {
        res.converged = true;
        break;
      }
      for (std::size_t i = 0; i < n; ++i) {
        z[i] = r[i] / std::max(de[i], 1e-8);
      }
      const double rz_new = dot_vec(n, r.data(), z.data());
      const double beta = rz_new / rz_old;
      for (std::size_t i = 0; i < n; ++i) {
        p[i] = z[i] + beta * p[i];
      }
      rz_old = rz_new;
    }
  }

  // Transform Z_mo -> AO matrix C_o Z C_v^T +/- C_v Z^T C_o^T
  std::vector<double> tmp_uv(static_cast<std::size_t>(nao) * n_vir, 0.0);
  std::vector<double> m_ao(static_cast<std::size_t>(nao) * nao, 0.0);
  gemm(nao, n_vir, n_occ, 1.0, op.Co_host().data(), false, res.z_mo.data(), false, 0.0,
       tmp_uv.data(), opts.backend, GemmMode::FP64);
  gemm(nao, nao, n_vir, 1.0, tmp_uv.data(), false, op.Cv_host().data(), true, 0.0, m_ao.data(),
       opts.backend, GemmMode::FP64);

  res.z_ao.assign(static_cast<std::size_t>(nao) * nao, 0.0);
  const double sign = opts.symmetric_perturbation ? 1.0 : -1.0;
  for (int u = 0; u < nao; ++u) {
    for (int v = 0; v < nao; ++v) {
      res.z_ao[static_cast<std::size_t>(u) * nao + v] =
          m_ao[static_cast<std::size_t>(u) * nao + v] +
          sign * m_ao[static_cast<std::size_t>(v) * nao + u];
    }
  }
  return res;
}

std::array<double, 9> compute_static_polarizability(
    int nao, int nmo, int n_occ, const double* mo_energies_host, const double* mo_coeff_host,
    const std::array<std::vector<double>, 3>& ao_dipoles_host, const ResponseCallbacks& callbacks,
    const CphfOptions& opts) {
  const int n_vir = nmo - n_occ;
  const int n_ov = n_occ * n_vir;
  std::vector<double> Co(static_cast<std::size_t>(nao) * n_occ);
  std::vector<double> Cv(static_cast<std::size_t>(nao) * n_vir);
  for (int u = 0; u < nao; ++u) {
    for (int i = 0; i < n_occ; ++i)
      Co[static_cast<std::size_t>(u) * n_occ + i] = mo_coeff_host[static_cast<std::size_t>(u) * nmo + i];
    for (int a = 0; a < n_vir; ++a)
      Cv[static_cast<std::size_t>(u) * n_vir + a] =
          mo_coeff_host[static_cast<std::size_t>(u) * nmo + (n_occ + a)];
  }

  std::array<std::vector<double>, 3> rhs_mu;
  std::vector<double> tmp_uv(static_cast<std::size_t>(nao) * n_vir, 0.0);
  for (int c = 0; c < 3; ++c) {
    rhs_mu[c].assign(static_cast<std::size_t>(n_ov), 0.0);
    gemm(nao, n_vir, nao, 2.0, ao_dipoles_host[c].data(), false, Cv.data(), false, 0.0,
         tmp_uv.data(), opts.backend, GemmMode::FP64);
    gemm(n_occ, n_vir, nao, 1.0, Co.data(), true, tmp_uv.data(), false, 0.0, rhs_mu[c].data(),
         opts.backend, GemmMode::FP64);
  }

  CphfOptions sym_opts = opts;
  sym_opts.symmetric_perturbation = true;
  std::array<CphfResult, 3> sol;
  for (int c = 0; c < 3; ++c) {
    sol[c] = solve_cphf(nao, nmo, n_occ, mo_energies_host, mo_coeff_host, rhs_mu[c].data(),
                        callbacks, sym_opts);
  }

  // \alpha_{ab} = 4 \mu_a^T (A + B)^{-1} \mu_b = (2 \mu_a)^T Z_b
  std::array<double, 9> alpha{};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      alpha[a * 3 + b] =
          dot_vec(static_cast<std::size_t>(n_ov), rhs_mu[a].data(), sol[b].z_mo.data());
    }
  }
  return alpha;
}

// ===========================================================================
// 4. Molecular Nuclear Hessian & Vibrational Harmonic Frequency Analysis
// ===========================================================================

double standard_atomic_mass_amu(int atomic_number) {
  // Exact PySCF `pyscf.data.elements.MASSES` table (Z = 0..86) so harmonic_analysis
  // frequencies match PySCF `thermo.harmonic_analysis(mol, hess)` to < 0.001 cm^{-1}!
  static constexpr double kMasses[] = {
      0.0,          1.008,        4.002602,     6.94,        9.0121831,    10.81,
      12.011,       14.007,       15.999,       18.998403163, 20.1797,     22.98976928,
      24.305,       26.9815385,   28.085,       30.973761998, 32.06,       35.45,
      39.948,       39.0983,      40.078,       44.955908,   47.867,       50.9415,
      51.9961,      54.938044,    55.845,       58.933194,   58.6934,      63.546,
      65.38,        69.723,       72.63,        74.921595,   78.971,       79.904,
      83.798,       85.4678,      87.62,        88.90584,    91.224,       92.90637,
      95.95,        97.90721,     101.07,       102.9055,    106.42,       107.8682,
      112.414,      114.818,      118.71,       121.76,      127.6,        126.90447,
      131.293,      132.90545196, 137.327,      138.90547,   140.116,      140.90766,
      144.242,      144.91276,    150.36,       151.964,     157.25,       158.92535,
      162.5,        164.93033,    167.259,      168.93422,   173.054,      174.9668,
      178.49,       180.94788,    183.84,       186.207,     190.23,       192.217,
      195.084,      196.966569,   200.592,      204.38,      207.2,        208.9804,
      208.98243,    209.98715,    222.01758,
  };
  constexpr int kMaxZ = static_cast<int>(sizeof(kMasses) / sizeof(kMasses[0])) - 1;
  if (atomic_number <= 0 || atomic_number > kMaxZ) {
    throw std::invalid_argument("standard_atomic_mass_amu: unsupported atomic number " +
                                std::to_string(atomic_number));
  }
  return kMasses[atomic_number];
}

VibrationalResult analyze_hessian(std::span<const Atom> atoms, const double* hessian_3n_3n_host,
                                  std::span<const double> custom_masses_amu,
                                  const HessianOptions& opts) {
  const auto t0 = SteadyClock::now();
  const int natom = static_cast<int>(atoms.size());
  if (natom <= 0 || hessian_3n_3n_host == nullptr) {
    throw std::invalid_argument("analyze_hessian: empty atoms or null hessian_3n_3n_host");
  }
  if (!custom_masses_amu.empty() &&
      static_cast<int>(custom_masses_amu.size()) != natom) {
    throw std::invalid_argument("analyze_hessian: custom_masses_amu length must equal natom");
  }

  const int n3 = 3 * natom;
  std::vector<double> masses(static_cast<std::size_t>(natom), 0.0);
  std::vector<double> sqrt_m(static_cast<std::size_t>(natom), 0.0);
  double total_mass = 0.0;
  std::array<double, 3> com = {0.0, 0.0, 0.0};

  for (int a = 0; a < natom; ++a) {
    const double m = custom_masses_amu.empty()
                         ? standard_atomic_mass_amu(atoms[a].atomic_number)
                         : custom_masses_amu[static_cast<std::size_t>(a)];
    if (!(m > 0.0)) {
      throw std::invalid_argument("analyze_hessian: atomic masses must be positive");
    }
    masses[static_cast<std::size_t>(a)] = m;
    sqrt_m[static_cast<std::size_t>(a)] = std::sqrt(m);
    total_mass += m;
    for (int c = 0; c < 3; ++c) com[c] += m * atoms[a].center[c];
  }
  for (int c = 0; c < 3; ++c) com[c] /= total_mass;

  VibrationalResult res;
  res.natom = natom;
  res.hessian.assign(static_cast<std::size_t>(n3) * n3, 0.0);

  std::vector<double> H_mw(static_cast<std::size_t>(n3) * n3, 0.0);
  for (int i = 0; i < n3; ++i) {
    const int ai = i / 3;
    for (int j = 0; j < n3; ++j) {
      const int aj = j / 3;
      const double h_sym = 0.5 * (hessian_3n_3n_host[static_cast<std::size_t>(i) * n3 + j] +
                                  hessian_3n_3n_host[static_cast<std::size_t>(j) * n3 + i]);
      res.hessian[static_cast<std::size_t>(i) * n3 + j] = h_sym;
      H_mw[static_cast<std::size_t>(i) * n3 + j] =
          h_sym / (sqrt_m[static_cast<std::size_t>(ai)] * sqrt_m[static_cast<std::size_t>(aj)]);
    }
  }

  // Construct orthonormal basis Q_vib [n3 x n_vib] orthogonal to Eckart TR subspace
  std::vector<std::vector<double>> D_tr;
  if (opts.project_trans_rot) {
    // 3 translations
    for (int c = 0; c < 3; ++c) {
      std::vector<double> t(static_cast<std::size_t>(n3), 0.0);
      for (int a = 0; a < natom; ++a) {
        t[static_cast<std::size_t>(3 * a + c)] = sqrt_m[static_cast<std::size_t>(a)];
      }
      append_orthonormal_vector(n3, D_tr, std::move(t), opts.tr_svd_tol);
    }
    // 3 rotations around center of mass
    std::array<std::vector<double>, 3> rot = {
        std::vector<double>(static_cast<std::size_t>(n3), 0.0),
        std::vector<double>(static_cast<std::size_t>(n3), 0.0),
        std::vector<double>(static_cast<std::size_t>(n3), 0.0),
    };
    for (int a = 0; a < natom; ++a) {
      const double sm = sqrt_m[static_cast<std::size_t>(a)];
      const double rx = atoms[a].center[0] - com[0];
      const double ry = atoms[a].center[1] - com[1];
      const double rz = atoms[a].center[2] - com[2];
      rot[0][static_cast<std::size_t>(3 * a + 1)] = -sm * rz;
      rot[0][static_cast<std::size_t>(3 * a + 2)] = +sm * ry;
      rot[1][static_cast<std::size_t>(3 * a + 0)] = +sm * rz;
      rot[1][static_cast<std::size_t>(3 * a + 2)] = -sm * rx;
      rot[2][static_cast<std::size_t>(3 * a + 0)] = -sm * ry;
      rot[2][static_cast<std::size_t>(3 * a + 1)] = +sm * rx;
    }
    for (int c = 0; c < 3; ++c) {
      append_orthonormal_vector(n3, D_tr, std::move(rot[c]), opts.tr_svd_tol);
    }
  }

  res.n_trans_rot = static_cast<int>(D_tr.size());
  res.n_vib = n3 - res.n_trans_rot;
  if (res.n_vib <= 0) {
    res.wall_time_s = elapsed_sec(t0, SteadyClock::now());
    return res;
  }

  // Complete D_tr to full orthonormal basis [D_tr, Q_vib] via 2-pass MGS on coordinate axes
  std::vector<std::vector<double>> full_basis = D_tr;
  full_basis.reserve(static_cast<std::size_t>(n3));
  for (int i = 0; i < n3 && static_cast<int>(full_basis.size()) < n3; ++i) {
    std::vector<double> e(static_cast<std::size_t>(n3), 0.0);
    e[static_cast<std::size_t>(i)] = 1.0;
    append_orthonormal_vector(n3, full_basis, std::move(e), 1e-7);
  }

  const int nvib = res.n_vib;
  std::vector<double> Q_vib(static_cast<std::size_t>(n3) * nvib, 0.0);
  for (int k = 0; k < nvib; ++k) {
    const auto& col = full_basis[static_cast<std::size_t>(res.n_trans_rot + k)];
    for (int i = 0; i < n3; ++i) {
      Q_vib[static_cast<std::size_t>(i) * nvib + k] = col[static_cast<std::size_t>(i)];
    }
  }

  const Backend diag_backend =
      backend_available(opts.backend) ? opts.backend : Backend::Host;
  std::vector<double> tmp_3n_v(static_cast<std::size_t>(n3) * nvib, 0.0);
  std::vector<double> H_vib(static_cast<std::size_t>(nvib) * nvib, 0.0);
  gemm(n3, nvib, n3, 1.0, H_mw.data(), false, Q_vib.data(), false, 0.0, tmp_3n_v.data(),
       diag_backend);
  gemm(nvib, nvib, n3, 1.0, Q_vib.data(), true, tmp_3n_v.data(), false, 0.0, H_vib.data(),
       diag_backend);
  for (int i = 0; i < nvib; ++i) {
    for (int j = i + 1; j < nvib; ++j) {
      const double avg = 0.5 * (H_vib[static_cast<std::size_t>(i) * nvib + j] +
                                H_vib[static_cast<std::size_t>(j) * nvib + i]);
      H_vib[static_cast<std::size_t>(i) * nvib + j] = avg;
      H_vib[static_cast<std::size_t>(j) * nvib + i] = avg;
    }
  }

  std::vector<double> evals(static_cast<std::size_t>(nvib), 0.0);
  std::vector<double> L_int(static_cast<std::size_t>(nvib) * nvib, 0.0);
  syevd(nvib, H_vib.data(), evals.data(), L_int.data(), diag_backend);

  // L_mw [n3 x nvib] = Q_vib [n3 x nvib] * L_int [nvib x nvib]
  std::vector<double> L_mw(static_cast<std::size_t>(n3) * nvib, 0.0);
  gemm(n3, nvib, nvib, 1.0, Q_vib.data(), false, L_int.data(), false, 0.0, L_mw.data(),
       diag_backend);

  const double conv_cm1 = kHartreeToCm1 / std::sqrt(kAmuToElectronMass);

  res.frequencies_cm1.resize(static_cast<std::size_t>(nvib));
  res.force_constants_au = evals;
  res.reduced_masses_amu.resize(static_cast<std::size_t>(nvib));
  res.normal_modes.assign(static_cast<std::size_t>(nvib) * n3, 0.0);
  res.mass_weighted_modes.assign(static_cast<std::size_t>(nvib) * n3, 0.0);

  double zpe_eh = 0.0;
  for (int k = 0; k < nvib; ++k) {
    const double lam = evals[static_cast<std::size_t>(k)];
    if (lam < 0.0) {
      res.frequencies_cm1[static_cast<std::size_t>(k)] = -conv_cm1 * std::sqrt(-lam);
      if (res.frequencies_cm1[static_cast<std::size_t>(k)] < -1.0) res.imaginary_count += 1;
    } else {
      const double freq_cm1 = conv_cm1 * std::sqrt(lam);
      res.frequencies_cm1[static_cast<std::size_t>(k)] = freq_cm1;
      zpe_eh += 0.5 * std::sqrt(lam / kAmuToElectronMass);
    }

    double sum_q2 = 0.0;
    for (int i = 0; i < n3; ++i) {
      const double l_ik = L_mw[static_cast<std::size_t>(i) * nvib + k];
      res.mass_weighted_modes[static_cast<std::size_t>(k) * n3 + i] = l_ik;
      const double q_ik = l_ik / sqrt_m[static_cast<std::size_t>(i / 3)];
      res.normal_modes[static_cast<std::size_t>(k) * n3 + i] = q_ik;
      sum_q2 += q_ik * q_ik;
    }
    const double red_mass = (sum_q2 > 1e-30) ? (1.0 / sum_q2) : 0.0;
    res.reduced_masses_amu[static_cast<std::size_t>(k)] = red_mass;
    const double inv_nrm = (sum_q2 > 1e-30) ? (1.0 / std::sqrt(sum_q2)) : 1.0;
    for (int i = 0; i < n3; ++i) {
      res.normal_modes[static_cast<std::size_t>(k) * n3 + i] *= inv_nrm;
    }
  }
  res.zero_point_energy_hartree = zpe_eh;
  res.wall_time_s = elapsed_sec(t0, SteadyClock::now());
  return res;
}

VibrationalResult compute_nuclear_hessian(
    std::span<const Atom> atoms,
    const std::function<std::vector<double>(std::span<const Atom>)>& grad_fn,
    std::span<const double> custom_masses_amu, const HessianOptions& opts) {
  const auto t0 = SteadyClock::now();
  const int natom = static_cast<int>(atoms.size());
  if (natom <= 0 || !grad_fn) {
    throw std::invalid_argument("compute_nuclear_hessian: empty atoms or invalid grad_fn");
  }
  if (!(opts.step_bohr > 0.0)) {
    throw std::invalid_argument("compute_nuclear_hessian: step_bohr must be positive");
  }

  const int n3 = 3 * natom;
  const double delta = opts.step_bohr;
  const double inv_2delta = 0.5 / delta;
  std::vector<double> raw_hess(static_cast<std::size_t>(n3) * n3, 0.0);

  for (int p = 0; p < n3; ++p) {
    const int ia = p / 3;
    const int ic = p % 3;

    std::vector<Atom> atoms_p(atoms.begin(), atoms.end());
    std::vector<Atom> atoms_m(atoms.begin(), atoms.end());
    atoms_p[static_cast<std::size_t>(ia)].center[ic] += delta;
    atoms_m[static_cast<std::size_t>(ia)].center[ic] -= delta;

    const std::vector<double> gp = grad_fn(atoms_p);
    const std::vector<double> gm = grad_fn(atoms_m);
    if (static_cast<int>(gp.size()) != n3 || static_cast<int>(gm.size()) != n3) {
      throw std::runtime_error("compute_nuclear_hessian: grad_fn returned wrong size");
    }

    for (int q = 0; q < n3; ++q) {
      raw_hess[static_cast<std::size_t>(p) * n3 + q] =
          (gp[static_cast<std::size_t>(q)] - gm[static_cast<std::size_t>(q)]) * inv_2delta;
    }
  }

  VibrationalResult res = analyze_hessian(atoms, raw_hess.data(), custom_masses_amu, opts);
  res.wall_time_s = elapsed_sec(t0, SteadyClock::now());
  return res;
}

VibrationalResult compute_molecular_hessian(
    const Basis& basis, std::span<const Atom> atoms, int n_elec, WorkflowOptions wf_opts,
    std::span<const double> custom_masses_amu, const HessianOptions& hess_opts,
    std::span<const double> external_grid_coords,
    std::span<const double> external_grid_weights) {
  const int natom = static_cast<int>(atoms.size());
  // Map each shell in `basis` to its parent atom at equilibrium R_0
  std::vector<int> shell_parent(basis.nshell(), -1);
  for (std::size_t s = 0; s < basis.nshell(); ++s) {
    const auto& sc = basis.shell(s).center;
    for (int a = 0; a < natom; ++a) {
      const double dx = sc[0] - atoms[a].center[0];
      const double dy = sc[1] - atoms[a].center[1];
      const double dz = sc[2] - atoms[a].center[2];
      if (dx * dx + dy * dy + dz * dz < 1e-10) {
        shell_parent[s] = a;
        break;
      }
    }
  }

  // Note: `Basis` constructor normalizes raw coefficients; to reconstruct a displaced `Basis`
  // with identical contracted primitives without re-normalizing already-normalized coefficients,
  // copy `basis` and shift shell centers via unit-primitive scaling? WAIT:
  // Let's check `Basis` constructor in `src/basis.cpp`!
  // Let's make sure shifting a shell center preserves exact radial normalization!
  // Wait: radial normalization in `Basis` ONLY depends on `(l, exponents, coefficients)` —Wait:
  // if `Basis(shells)` expects unnormalized `bas_ctr_coeff`, passing `basis.shell(s).coefficients`
  // into `Basis(...)` a SECOND time would re-multiply by `r^l` primitive norm unless we invert
  // or notice that radial normalization is 100% independent of `center`!
  // Let's check `src/basis.cpp` before finalizing this function!

  // Tighten SCF convergence for accurate numerical derivatives of analytical gradients
  wf_opts.scf.backend = hess_opts.backend;
  wf_opts.scf.conv_tol = std::min(wf_opts.scf.conv_tol, 1e-11);
  wf_opts.scf.diis_tol = std::min(wf_opts.scf.diis_tol, 1e-8);
  wf_opts.scf.dynamic_precision = false;
  wf_opts.scf.incremental_fock = false;
  wf_opts.grid.backend = hess_opts.backend;
  wf_opts.xc_options.backend = hess_opts.backend;

  std::vector<double> fixed_coords(external_grid_coords.begin(), external_grid_coords.end());
  std::vector<double> fixed_weights(external_grid_weights.begin(), external_grid_weights.end());
  if (hess_opts.fixed_xc_grid && fixed_coords.empty() && !is_hf_xc_name(wf_opts.xc)) {
    std::vector<XcAtom> xc_atoms;
    xc_atoms.reserve(atoms.size());
    for (const Atom& a : atoms) xc_atoms.push_back(XcAtom{a.atomic_number, a.center});
    GridOptions gopts = wf_opts.grid;
    if (gopts.level < 4) gopts.level = 4;
    MolecularGrid eq_grid(xc_atoms, basis, gopts);
    fixed_coords = eq_grid.coords();
    fixed_weights = eq_grid.weights();
  }

  // Reconstruct raw unnormalized contraction coefficients so Basis(displaced_shells) produces
  // bit-identical `coefficients` to `basis.shell(s).coefficients`!
  // Recall for primitive r^l exp(-a r^2), \int_0^\infty (r^l e^{-a r^2})^2 r^2 dr:
  // If we pass a single primitive `[a]` with coeff `1.0` to `Basis({Shell{l, {0,0,0}, {a}, {1.0}}})`,
  // `Basis` turns `1.0` into `N_prim(l, a)`! Dividing `basis.shell(s).coefficients[k]` by `N_prim(l, a_k)`
  // recovers the exact input coefficient (up to overall shell norm 1, which `Basis` normalizes to 1)!
  std::vector<Shell> raw_template_shells;
  raw_template_shells.reserve(basis.nshell());
  for (std::size_t s = 0; s < basis.nshell(); ++s) {
    const Shell& sh = basis.shell(s);
    Shell raw;
    raw.l = sh.l;
    raw.center = sh.center;
    raw.exponents = sh.exponents;
    raw.coefficients.resize(sh.coefficients.size());
    for (std::size_t k = 0; k < sh.exponents.size(); ++k) {
      Basis single({Shell{sh.l, {0.0, 0.0, 0.0}, {sh.exponents[k]}, {1.0}}});
      raw.coefficients[k] = sh.coefficients[k] / single.shell(0).coefficients[0];
    }
    raw_template_shells.push_back(std::move(raw));
  }

  auto grad_fn = [&](std::span<const Atom> disp_atoms) -> std::vector<double> {
    std::vector<Shell> disp_shells = raw_template_shells;
    for (std::size_t s = 0; s < disp_shells.size(); ++s) {
      const int pa = shell_parent[s];
      if (pa >= 0) {
        for (int c = 0; c < 3; ++c) {
          const double shift = disp_atoms[static_cast<std::size_t>(pa)].center[c] -
                               atoms[static_cast<std::size_t>(pa)].center[c];
          disp_shells[s].center[c] = raw_template_shells[s].center[c] + shift;
        }
      }
    }
    Basis disp_basis(std::move(disp_shells));
    std::vector<Atom> at_vec(disp_atoms.begin(), disp_atoms.end());
    MolecularScf mscf(std::move(disp_basis), std::move(at_vec), n_elec, wf_opts, std::nullopt,
                      fixed_coords, fixed_weights);
    const WorkflowResult wres = mscf.run();
    if (!wres.scf.converged) {
      throw std::runtime_error("compute_molecular_hessian: displaced SCF failed to converge");
    }
    return mscf.compute_gradient(wres.scf).total_grad;
  };

  return compute_nuclear_hessian(atoms, grad_fn, custom_masses_amu, hess_opts);
}

}  // namespace gpu4dft

// ===========================================================================
// Standalone C ABI Entry Points (`gpu4dft_resp_c_*`)
// ===========================================================================

namespace {

gpu4dft::Backend resp_c_backend(int b) {
  return (b == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
}

gpu4dft::GemmMode resp_c_gemm_mode(int m) {
  if (m == 1) return gpu4dft::GemmMode::FP32;
  if (m == 2) return gpu4dft::GemmMode::OzakiInt8;
  return gpu4dft::GemmMode::FP64;
}

gpu4dft::Basis unpack_c_basis(int nshell, const int* shell_l, const double* shell_centers_3n,
                              const int* prim_offsets, const double* prim_exps,
                              const double* prim_coeffs) {
  std::vector<gpu4dft::Shell> shells;
  shells.reserve(static_cast<std::size_t>(nshell));
  for (int s = 0; s < nshell; ++s) {
    gpu4dft::Shell sh;
    sh.l = shell_l[s];
    sh.center = {shell_centers_3n[3 * s + 0], shell_centers_3n[3 * s + 1],
                 shell_centers_3n[3 * s + 2]};
    const int p0 = prim_offsets[s];
    const int p1 = prim_offsets[s + 1];
    sh.exponents.assign(prim_exps + p0, prim_exps + p1);
    sh.coefficients.assign(prim_coeffs + p0, prim_coeffs + p1);
    shells.push_back(std::move(sh));
  }
  return gpu4dft::Basis(std::move(shells));
}

std::vector<gpu4dft::Atom> unpack_c_atoms(int natom, const int* atom_Z,
                                          const double* atom_centers_3n) {
  std::vector<gpu4dft::Atom> atoms;
  atoms.reserve(static_cast<std::size_t>(natom));
  for (int a = 0; a < natom; ++a) {
    atoms.push_back(gpu4dft::Atom{
        atom_Z[a],
        {atom_centers_3n[3 * a + 0], atom_centers_3n[3 * a + 1], atom_centers_3n[3 * a + 2]},
    });
  }
  return atoms;
}

}  // namespace

GPU4DFT_RESP_C_EXPORT int gpu4dft_resp_c_cuda_available() {
  return gpu4dft::backend_available(gpu4dft::Backend::Cuda) ? 1 : 0;
}

GPU4DFT_RESP_C_EXPORT int gpu4dft_resp_c_dipole_integrals(
    int nshell, const int* shell_l, const double* shell_centers_3n, const int* prim_offsets,
    const double* prim_exps, const double* prim_coeffs, const double* origin3, int backend,
    double* out_3_nao2) {
  try {
    const gpu4dft::Basis basis =
        unpack_c_basis(nshell, shell_l, shell_centers_3n, prim_offsets, prim_exps, prim_coeffs);
    std::array<double, 3> orig = {0.0, 0.0, 0.0};
    if (origin3 != nullptr) orig = {origin3[0], origin3[1], origin3[2]};
    const auto mats = gpu4dft::dipole_matrices(basis, orig, resp_c_backend(backend));
    const std::size_t n2 = static_cast<std::size_t>(basis.nao()) * basis.nao();
    for (int c = 0; c < 3; ++c) {
      std::copy(mats[c].begin(), mats[c].end(), out_3_nao2 + static_cast<std::size_t>(c) * n2);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

using gpu4dft_resp_fxc_cb_t = void (*)(const double* sym_dm_host, double* fxc_out_host,
                                       void* user_data);

GPU4DFT_RESP_C_EXPORT int gpu4dft_resp_c_run_response_eri(
    int nao, int nmo, int n_occ, const double* mo_energies, const double* mo_coeff,
    const double* eri_host, const double* ao_dipoles_3_nao2, int backend, int method,
    int gemm_mode, int ozaki_slices, int dynamic_prec, int n_states, double alpha_x,
    double conv_tol, gpu4dft_resp_fxc_cb_t fxc_cb, void* fxc_user, double* energies_eh_out,
    double* osc_out, double* tdip_3n_out, double* x_vecs_out, double* y_vecs_out,
    double* stats4_out) {
  try {
    gpu4dft::ResponseOptions opts;
    opts.backend = resp_c_backend(backend);
    opts.method = (method == 0) ? gpu4dft::ResponseMethod::TDA : gpu4dft::ResponseMethod::TDDFT;
    opts.gemm_mode = resp_c_gemm_mode(gemm_mode);
    opts.ozaki_slices = ozaki_slices;
    opts.dynamic_precision = (dynamic_prec != 0);
    opts.n_states = n_states;
    opts.exact_exchange_fraction = alpha_x;
    opts.conv_tol = conv_tol;

    const bool need_k = (std::abs(alpha_x) > 0.0);
    gpu4dft::ResponseCallbacks cb = gpu4dft::make_in_core_eri_response_callbacks(
        nao, eri_host, opts.backend, need_k, opts.gemm_mode, opts.ozaki_slices);
    if (fxc_cb != nullptr) {
      cb.fxc_device_resident = false;
      cb.build_fxc = [fxc_cb, fxc_user](const double* sym_dm, double* fxc_out, void*) {
        fxc_cb(sym_dm, fxc_out, fxc_user);
      };
    }

    std::array<std::vector<double>, 3> dip_arr;
    const std::array<std::vector<double>, 3>* dip_ptr = nullptr;
    if (ao_dipoles_3_nao2 != nullptr) {
      const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
      for (int c = 0; c < 3; ++c) {
        dip_arr[c].assign(ao_dipoles_3_nao2 + static_cast<std::size_t>(c) * n2,
                          ao_dipoles_3_nao2 + static_cast<std::size_t>(c + 1) * n2);
      }
      dip_ptr = &dip_arr;
    }

    gpu4dft::LinearResponseDriver drv(opts);
    const gpu4dft::ResponseResult res =
        drv.run(nao, nmo, n_occ, mo_energies, mo_coeff, cb, dip_ptr);

    const int nr = res.n_states;
    if (energies_eh_out) {
      std::copy(res.excitation_energies.begin(), res.excitation_energies.end(), energies_eh_out);
    }
    if (osc_out) {
      std::copy(res.oscillator_strengths.begin(), res.oscillator_strengths.end(), osc_out);
    }
    if (tdip_3n_out) {
      for (int k = 0; k < nr; ++k) {
        tdip_3n_out[3 * k + 0] = res.transition_dipoles[static_cast<std::size_t>(k)][0];
        tdip_3n_out[3 * k + 1] = res.transition_dipoles[static_cast<std::size_t>(k)][1];
        tdip_3n_out[3 * k + 2] = res.transition_dipoles[static_cast<std::size_t>(k)][2];
      }
    }
    if (x_vecs_out) {
      std::copy(res.x_vectors.begin(), res.x_vectors.end(), x_vecs_out);
    }
    if (y_vecs_out) {
      std::copy(res.y_vectors.begin(), res.y_vectors.end(), y_vecs_out);
    }
    if (stats4_out) {
      stats4_out[0] = res.converged ? 1.0 : 0.0;
      stats4_out[1] = static_cast<double>(res.iterations);
      stats4_out[2] = static_cast<double>(res.matvec_count);
      stats4_out[3] = res.wall_time_s;
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_RESP_C_EXPORT int gpu4dft_resp_c_run_molecular_response(
    int nshell, const int* shell_l, const double* shell_centers_3n, const int* prim_offsets,
    const double* prim_exps, const double* prim_coeffs, int natom, const int* atom_Z,
    const double* atom_centers_3n, int n_elec, const char* xc_name, const double* grid_coords_3g,
    const double* grid_weights_g, int ngrid, int backend, int method, int gemm_mode,
    int ozaki_slices, int n_states, double conv_tol, double* e_scf_out, double* energies_eh_out,
    double* energies_ev_out, double* osc_out, double* tdip_3n_out, double* stats4_out) {
  try {
    const gpu4dft::Basis basis =
        unpack_c_basis(nshell, shell_l, shell_centers_3n, prim_offsets, prim_exps, prim_coeffs);
    const std::vector<gpu4dft::Atom> atoms = unpack_c_atoms(natom, atom_Z, atom_centers_3n);
    const std::string xc = (xc_name != nullptr) ? std::string(xc_name) : "hf";
    const gpu4dft::Backend be = resp_c_backend(backend);

    std::span<const double> g_coords;
    std::span<const double> g_weights;
    if (ngrid > 0 && grid_coords_3g != nullptr && grid_weights_g != nullptr) {
      g_coords = std::span<const double>(grid_coords_3g, static_cast<std::size_t>(ngrid) * 3);
      g_weights = std::span<const double>(grid_weights_g, static_cast<std::size_t>(ngrid));
    }

    gpu4dft::WorkflowOptions wf_opts;
    wf_opts.xc = xc;
    wf_opts.jk_mode = gpu4dft::JKMode::Direct;
    wf_opts.scf.backend = be;
    wf_opts.scf.solver = gpu4dft::DensitySolver::Diag;
    wf_opts.scf.conv_tol = 1e-13;
    wf_opts.scf.diis_tol = 1e-10;
    wf_opts.scf.dynamic_precision = false;
    wf_opts.grid.backend = be;
    wf_opts.grid.level = 4;
    wf_opts.xc_options.backend = be;

    const gpu4dft::WorkflowResult scf_wres =
        gpu4dft::run_scf(basis, atoms, n_elec, wf_opts, {}, nullptr, g_coords, g_weights);
    if (!scf_wres.scf.converged) return -2;
    if (e_scf_out) *e_scf_out = scf_wres.scf.total_energy;

    gpu4dft::ResponseOptions ropts;
    ropts.backend = be;
    ropts.method = (method == 0) ? gpu4dft::ResponseMethod::TDA : gpu4dft::ResponseMethod::TDDFT;
    ropts.gemm_mode = resp_c_gemm_mode(gemm_mode);
    ropts.ozaki_slices = ozaki_slices;
    ropts.n_states = n_states;
    ropts.conv_tol = conv_tol;

    const gpu4dft::ResponseResult res = gpu4dft::run_linear_response(
        basis, atoms, n_elec / 2, scf_wres.scf, xc, ropts, g_coords, g_weights, wf_opts.grid);

    const int nr = res.n_states;
    if (energies_eh_out) {
      std::copy(res.excitation_energies.begin(), res.excitation_energies.end(), energies_eh_out);
    }
    if (energies_ev_out) {
      std::copy(res.excitation_energies_ev.begin(), res.excitation_energies_ev.end(),
                energies_ev_out);
    }
    if (osc_out) {
      std::copy(res.oscillator_strengths.begin(), res.oscillator_strengths.end(), osc_out);
    }
    if (tdip_3n_out) {
      for (int k = 0; k < nr; ++k) {
        tdip_3n_out[3 * k + 0] = res.transition_dipoles[static_cast<std::size_t>(k)][0];
        tdip_3n_out[3 * k + 1] = res.transition_dipoles[static_cast<std::size_t>(k)][1];
        tdip_3n_out[3 * k + 2] = res.transition_dipoles[static_cast<std::size_t>(k)][2];
      }
    }
    if (stats4_out) {
      stats4_out[0] = res.converged ? 1.0 : 0.0;
      stats4_out[1] = static_cast<double>(res.iterations);
      stats4_out[2] = static_cast<double>(res.matvec_count);
      stats4_out[3] = res.wall_time_s;
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_RESP_C_EXPORT int gpu4dft_resp_c_polarizability_eri(
    int nao, int nmo, int n_occ, const double* mo_energies, const double* mo_coeff,
    const double* eri_host, const double* ao_dipoles_3_nao2, int backend, double alpha_x,
    double conv_tol, gpu4dft_resp_fxc_cb_t fxc_cb, void* fxc_user, double* polar_9_out) {
  try {
    const gpu4dft::Backend be = resp_c_backend(backend);
    const bool need_k = (std::abs(alpha_x) > 0.0);
    gpu4dft::ResponseCallbacks cb =
        gpu4dft::make_in_core_eri_response_callbacks(nao, eri_host, be, need_k);
    if (fxc_cb != nullptr) {
      cb.fxc_device_resident = false;
      cb.build_fxc = [fxc_cb, fxc_user](const double* sym_dm, double* fxc_out, void*) {
        fxc_cb(sym_dm, fxc_out, fxc_user);
      };
    }

    const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
    std::array<std::vector<double>, 3> dip_arr;
    for (int c = 0; c < 3; ++c) {
      dip_arr[c].assign(ao_dipoles_3_nao2 + static_cast<std::size_t>(c) * n2,
                        ao_dipoles_3_nao2 + static_cast<std::size_t>(c + 1) * n2);
    }

    gpu4dft::CphfOptions copts;
    copts.backend = be;
    copts.exact_exchange_fraction = alpha_x;
    copts.conv_tol = conv_tol;

    const auto pol = gpu4dft::compute_static_polarizability(nao, nmo, n_occ, mo_energies, mo_coeff,
                                                            dip_arr, cb, copts);
    std::copy(pol.begin(), pol.end(), polar_9_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_RESP_C_EXPORT int gpu4dft_resp_c_compute_molecular_hessian(
    int nshell, const int* shell_l, const double* shell_centers_3n, const int* prim_offsets,
    const double* prim_exps, const double* prim_coeffs, int natom, const int* atom_Z,
    const double* atom_centers_3n, const double* custom_masses_amu, int n_elec,
    const char* xc_name, const double* grid_coords_3g, const double* grid_weights_g, int ngrid,
    int backend, double step_bohr, int fixed_xc_grid, double* hessian_out, int* nvib_out,
    double* freqs_cm1_out, double* red_masses_out, double* modes_out, double* zpe_out) {
  try {
    const gpu4dft::Basis basis =
        unpack_c_basis(nshell, shell_l, shell_centers_3n, prim_offsets, prim_exps, prim_coeffs);
    const std::vector<gpu4dft::Atom> atoms = unpack_c_atoms(natom, atom_Z, atom_centers_3n);
    const gpu4dft::Backend be = resp_c_backend(backend);

    std::span<const double> masses_span;
    if (custom_masses_amu != nullptr) {
      masses_span = std::span<const double>(custom_masses_amu, static_cast<std::size_t>(natom));
    }
    std::span<const double> g_coords;
    std::span<const double> g_weights;
    if (ngrid > 0 && grid_coords_3g != nullptr && grid_weights_g != nullptr) {
      g_coords = std::span<const double>(grid_coords_3g, static_cast<std::size_t>(ngrid) * 3);
      g_weights = std::span<const double>(grid_weights_g, static_cast<std::size_t>(ngrid));
    }

    gpu4dft::WorkflowOptions wf_opts;
    wf_opts.xc = (xc_name != nullptr) ? std::string(xc_name) : "hf";
    wf_opts.jk_mode = gpu4dft::JKMode::Direct;
    wf_opts.scf.backend = be;
    wf_opts.grid.backend = be;
    wf_opts.grid.level = 4;
    wf_opts.grid.partition = gpu4dft::PartitionScheme::Becke;
    wf_opts.xc_options.backend = be;

    gpu4dft::HessianOptions hopts;
    hopts.backend = be;
    hopts.step_bohr = step_bohr;
    hopts.project_trans_rot = true;
    hopts.fixed_xc_grid = (fixed_xc_grid != 0);

    const gpu4dft::VibrationalResult res = gpu4dft::compute_molecular_hessian(
        basis, atoms, n_elec, wf_opts, masses_span, hopts, g_coords, g_weights);

    if (hessian_out) std::copy(res.hessian.begin(), res.hessian.end(), hessian_out);
    if (nvib_out) *nvib_out = res.n_vib;
    if (freqs_cm1_out) {
      std::copy(res.frequencies_cm1.begin(), res.frequencies_cm1.end(), freqs_cm1_out);
    }
    if (red_masses_out) {
      std::copy(res.reduced_masses_amu.begin(), res.reduced_masses_amu.end(), red_masses_out);
    }
    if (modes_out) {
      std::copy(res.normal_modes.begin(), res.normal_modes.end(), modes_out);
    }
    if (zpe_out) *zpe_out = res.zero_point_energy_hartree;
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_RESP_C_EXPORT int gpu4dft_resp_c_analyze_hessian(
    int natom, const int* atom_Z, const double* atom_centers_3n, const double* custom_masses_amu,
    const double* hessian_in, int backend, int* nvib_out, double* freqs_cm1_out,
    double* red_masses_out, double* modes_out, double* zpe_out) {
  try {
    const std::vector<gpu4dft::Atom> atoms = unpack_c_atoms(natom, atom_Z, atom_centers_3n);
    std::span<const double> masses_span;
    if (custom_masses_amu != nullptr) {
      masses_span = std::span<const double>(custom_masses_amu, static_cast<std::size_t>(natom));
    }
    gpu4dft::HessianOptions hopts;
    hopts.backend = resp_c_backend(backend);
    hopts.project_trans_rot = true;

    const gpu4dft::VibrationalResult res =
        gpu4dft::analyze_hessian(atoms, hessian_in, masses_span, hopts);
    if (nvib_out) *nvib_out = res.n_vib;
    if (freqs_cm1_out) {
      std::copy(res.frequencies_cm1.begin(), res.frequencies_cm1.end(), freqs_cm1_out);
    }
    if (red_masses_out) {
      std::copy(res.reduced_masses_amu.begin(), res.reduced_masses_amu.end(), red_masses_out);
    }
    if (modes_out) {
      std::copy(res.normal_modes.begin(), res.normal_modes.end(), modes_out);
    }
    if (zpe_out) *zpe_out = res.zero_point_energy_hartree;
    return 0;
  } catch (...) {
    return -1;
  }
}
