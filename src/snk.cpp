#include "gpu4dft/snk.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu4dft {

namespace snk_detail {

inline SnkPairTable build_snk_pair_table(const Basis& basis,
                                         const detail::FlatBasisData& flat_b,
                                         const std::vector<double>& s_exact,
                                         double pair_cutoff) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  constexpr double kSqrtPiOver2 = 0.88622692545275801365;

  SnkPairTable tbl;
  const int nsh = flat_b.nshell;
  const std::size_t nao = static_cast<std::size_t>(flat_b.nao);
  if (nsh == 0 || nao == 0) return tbl;

  const detail::FlatBasisView bview = flat_b.view();
  const std::vector<double> shell_radii = MolecularGrid::compute_shell_radii(basis, 1e-15);

  // Exact Cauchy-Schwarz per-shell bounds: V_diag_max[is] and S_diag_sqrt[is]
  std::vector<double> v_diag_max(static_cast<std::size_t>(nsh), 0.0);
  std::vector<double> s_diag_sqrt(static_cast<std::size_t>(nsh), 1.0);
  for (int is = 0; is < nsh; ++is) {
    const int nm = 2 * bview.shell_l[is] + 1;
    const int ao0 = bview.shell_ao_offset[is];
    double v_blk[81];
    eval_shell_pair_esp(bview, is, is,
                        bview.shell_center[3 * is + 0],
                        bview.shell_center[3 * is + 1],
                        bview.shell_center[3 * is + 2],
                        0.0, v_blk);
    double vmax = 0.0;
    double smax = 0.0;
    for (int m = 0; m < nm; ++m) {
      vmax = std::max(vmax, std::abs(v_blk[m * nm + m]));
      if (!s_exact.empty()) {
        const std::size_t u = static_cast<std::size_t>(ao0 + m);
        smax = std::max(smax, std::sqrt(std::max(0.0, s_exact[u * nao + u])));
      } else {
        smax = 1.0;
      }
    }
    v_diag_max[static_cast<std::size_t>(is)] = vmax;
    s_diag_sqrt[static_cast<std::size_t>(is)] = smax;
  }

  for (int ia_raw = 0; ia_raw < nsh; ++ia_raw) {
    for (int ib_raw = 0; ib_raw <= ia_raw; ++ib_raw) {
      int ia = ia_raw;
      int ib = ib_raw;
      if (bview.shell_l[ia] < bview.shell_l[ib]) {
        std::swap(ia, ib);
      }

      const int la = bview.shell_l[ia];
      const int lb = bview.shell_l[ib];
      const int L = la + lb;
      const int nma = 2 * la + 1;
      const int nmb = 2 * lb + 1;
      const int ao_a = bview.shell_ao_offset[ia];
      const int ao_b = bview.shell_ao_offset[ib];

      const double ax = bview.shell_center[3 * ia + 0];
      const double ay = bview.shell_center[3 * ia + 1];
      const double az = bview.shell_center[3 * ia + 2];
      const double bx = bview.shell_center[3 * ib + 0];
      const double by = bview.shell_center[3 * ib + 1];
      const double bz = bview.shell_center[3 * ib + 2];
      const double rab2 =
          (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz);

      const int pa0 = bview.shell_prim_offset[ia];
      const int npa = bview.shell_prim_count[ia];
      const int pb0 = bview.shell_prim_offset[ib];
      const int npb = bview.shell_prim_count[ib];

      SphCartTerm ta[28], tb[28];
      const int nta = sph_cart_terms(la, ta);
      const int ntb = sph_cart_terms(lb, tb);
      const int n_tet = tet_count(L);
      const int h_block_len = nma * nmb * n_tet;

      const int p_start = static_cast<int>(tbl.prims.size());
      double sum_w = 0.0;
      double sum_px = 0.0, sum_py = 0.0, sum_pz = 0.0;
      double p_min = std::numeric_limits<double>::infinity();

      std::vector<double> asymp_accum;
      const bool concentric = (rab2 < 1e-12);
      if (concentric && L <= 4) {
        asymp_accum.assign(static_cast<std::size_t>(h_block_len), 0.0);
      }

      double Ex[5][5][9], Ey[5][5][9], Ez[5][5][9];

      for (int ka = 0; ka < npa; ++ka) {
        const double alpha = bview.prim_exp[pa0 + ka];
        const double ca = bview.prim_coeff[pa0 + ka];
        for (int kb = 0; kb < npb; ++kb) {
          const double beta = bview.prim_exp[pb0 + kb];
          const double cb = bview.prim_coeff[pb0 + kb];
          const double p = alpha + beta;
          const double inv_p = 1.0 / p;
          const double mu = alpha * beta * inv_p;
          if (mu * rab2 > 42.0) continue;
          const double pre = (kTwoPi * inv_p) * ca * cb * std::exp(-mu * rab2);
          if (std::abs(pre) < pair_cutoff) continue;

          const double px = (alpha * ax + beta * bx) * inv_p;
          const double py = (alpha * ay + beta * by) * inv_p;
          const double pz = (alpha * az + beta * bz) * inv_p;

          SnkPrimPair pp;
          pp.px = px;
          pp.py = py;
          pp.pz = pz;
          pp.p = p;
          pp.pre = pre;

          if (L <= 4) {
            pp.h_offset = static_cast<int>(tbl.h_coeffs.size());
            tbl.h_coeffs.resize(tbl.h_coeffs.size() + static_cast<std::size_t>(h_block_len), 0.0);
            double* h_dst = tbl.h_coeffs.data() + pp.h_offset;

            if (L == 0) {
              h_dst[0] = pre * ta[0].coeff * tb[0].coeff;
            } else {
              md_hermite_1d(la, lb, px - ax, px - bx, 0.5 * inv_p, Ex);
              md_hermite_1d(la, lb, py - ay, py - by, 0.5 * inv_p, Ey);
              md_hermite_1d(la, lb, pz - az, pz - bz, 0.5 * inv_p, Ez);
              for (int i_a = 0; i_a < nta; ++i_a) {
                const SphCartTerm& sa = ta[i_a];
                for (int i_b = 0; i_b < ntb; ++i_b) {
                  const SphCartTerm& sb = tb[i_b];
                  const double sc = pre * (sa.coeff * sb.coeff);
                  double* row = h_dst + (sa.m * nmb + sb.m) * n_tet;
                  const int mx = sa.ix + sb.ix;
                  const int my = sa.iy + sb.iy;
                  const int mz = sa.iz + sb.iz;
                  for (int t = 0; t <= mx; ++t) {
                    const double ex = Ex[sa.ix][sb.ix][t];
                    if (ex == 0.0) continue;
                    for (int u = 0; u <= my; ++u) {
                      const double exy = ex * Ey[sa.iy][sb.iy][u];
                      if (exy == 0.0) continue;
                      for (int v = 0; v <= mz; ++v) {
                        row[tet_index(t, u, v)] += sc * exy * Ez[sa.iz][sb.iz][v];
                      }
                    }
                  }
                }
              }
            }

            if (concentric) {
              const double asymp_scale = kSqrtPiOver2 / std::sqrt(p);
              for (int idx = 0; idx < h_block_len; ++idx) {
                asymp_accum[static_cast<std::size_t>(idx)] += asymp_scale * h_dst[idx];
              }
            }
          }

          const double w = std::abs(pre);
          sum_w += w;
          sum_px += w * px;
          sum_py += w * py;
          sum_pz += w * pz;
          p_min = std::min(p_min, p);
          tbl.prims.push_back(pp);
        }
      }

      const int p_cnt = static_cast<int>(tbl.prims.size()) - p_start;
      if (p_cnt == 0) continue;

      SnkShellPair sp;
      sp.ish_a = ia;
      sp.ish_b = ib;
      sp.la = la;
      sp.lb = lb;
      sp.ao_a = ao_a;
      sp.ao_b = ao_b;
      sp.nma = nma;
      sp.nmb = nmb;
      sp.prim_offset = p_start;
      sp.prim_count = p_cnt;
      sp.p_min = p_min;

      if (sum_w > 0.0) {
        sp.center[0] = sum_px / sum_w;
        sp.center[1] = sum_py / sum_w;
        sp.center[2] = sum_pz / sum_w;
      } else {
        sp.center[0] = 0.5 * (ax + bx);
        sp.center[1] = 0.5 * (ay + by);
        sp.center[2] = 0.5 * (az + bz);
      }

      const double da = std::sqrt((ax - sp.center[0]) * (ax - sp.center[0]) +
                                  (ay - sp.center[1]) * (ay - sp.center[1]) +
                                  (az - sp.center[2]) * (az - sp.center[2]));
      const double db = std::sqrt((bx - sp.center[0]) * (bx - sp.center[0]) +
                                  (by - sp.center[1]) * (by - sp.center[1]) +
                                  (bz - sp.center[2]) * (bz - sp.center[2]));
      // Product chi_a * chi_b is < 1e-15 outside BOTH shell_radii spheres -> intersection min!
      sp.extent_radius = std::min(da + shell_radii[static_cast<std::size_t>(ia)],
                                  db + shell_radii[static_cast<std::size_t>(ib)]);

      if (!s_exact.empty()) {
        if (L == 0) {
          sp.q_monopole = s_exact[static_cast<std::size_t>(ao_a) * nao +
                                  static_cast<std::size_t>(ao_b)];
        } else {
          double norm2 = 0.0;
          for (int ma = 0; ma < nma; ++ma) {
            for (int mb = 0; mb < nmb; ++mb) {
              const double sv = s_exact[static_cast<std::size_t>(ao_a + ma) * nao +
                                        static_cast<std::size_t>(ao_b + mb)];
              norm2 += sv * sv;
            }
          }
          sp.q_monopole = std::sqrt(norm2);
        }
      }

      sp.v_max_near = std::sqrt(v_diag_max[static_cast<std::size_t>(ia)] *
                                v_diag_max[static_cast<std::size_t>(ib)]);
      sp.q_far_bound = s_diag_sqrt[static_cast<std::size_t>(ia)] *
                       s_diag_sqrt[static_cast<std::size_t>(ib)];

      if (concentric && L <= 4) {
        sp.asymp_h_offset = static_cast<int>(tbl.h_coeffs.size());
        tbl.h_coeffs.insert(tbl.h_coeffs.end(), asymp_accum.begin(), asymp_accum.end());
      }

      tbl.pairs.push_back(sp);
    }
  }
  return tbl;
}

template <int L>
inline void eval_precomputed_pair_esp_host(const SnkShellPair& sp,
                                           const SnkPrimPair* prims,
                                           const double* h_coeffs,
                                           double gx, double gy, double gz,
                                           double* v_sph) {
  const int n_uv = sp.nma * sp.nmb;
  constexpr int n_tet = tet_count(L);

  if (sp.asymp_h_offset >= 0) {
    const double xpc = sp.center[0] - gx;
    const double ypc = sp.center[1] - gy;
    const double zpc = sp.center[2] - gz;
    const double rpc2 = xpc * xpc + ypc * ypc + zpc * zpc;
    if (sp.p_min * rpc2 > boys_asymp_threshold(L)) {
      const double inv_r = 1.0 / std::sqrt(rpc2);
      const double* M = h_coeffs + sp.asymp_h_offset;
      if constexpr (L == 0) {
        v_sph[0] = M[0] * inv_r;
        return;
      }
      const double inv_r2 = inv_r * inv_r;
      double d[5];
      d[0] = inv_r;
      if constexpr (L >= 1) d[1] = -d[0] * inv_r2;
      if constexpr (L >= 2) d[2] = -3.0 * d[1] * inv_r2;
      if constexpr (L >= 3) d[3] = -5.0 * d[2] * inv_r2;
      if constexpr (L >= 4) d[4] = -7.0 * d[3] * inv_r2;

      double R[n_tet];
      eval_hermite_coulomb_reg<L>(d, xpc, ypc, zpc, R);
      for (int uv = 0; uv < n_uv; ++uv) {
        const double* row = M + uv * n_tet;
        double acc = 0.0;
        for (int t = 0; t < n_tet; ++t) acc += row[t] * R[t];
        v_sph[uv] = acc;
      }
      return;
    }
  }

  for (int uv = 0; uv < n_uv; ++uv) v_sph[uv] = 0.0;
  for (int kp = 0; kp < sp.prim_count; ++kp) {
    const SnkPrimPair& pp = prims[sp.prim_offset + kp];
    const double xpc = pp.px - gx;
    const double ypc = pp.py - gy;
    const double zpc = pp.pz - gz;
    const double rpc2 = xpc * xpc + ypc * ypc + zpc * zpc;
    const double* H = h_coeffs + pp.h_offset;

    double F[5];
    boys_vec_hd(L, pp.p * rpc2, F);
    if constexpr (L == 0) {
      v_sph[0] += H[0] * F[0];
    } else {
      const double m2p = -2.0 * pp.p;
      double d[5];
      d[0] = F[0];
      if constexpr (L >= 1) d[1] = m2p * F[1];
      if constexpr (L >= 2) d[2] = (m2p * m2p) * F[2];
      if constexpr (L >= 3) d[3] = (m2p * m2p * m2p) * F[3];
      if constexpr (L >= 4) d[4] = (m2p * m2p) * (m2p * m2p) * F[4];

      double R[n_tet];
      eval_hermite_coulomb_reg<L>(d, xpc, ypc, zpc, R);
      for (int uv = 0; uv < n_uv; ++uv) {
        const double* row = H + uv * n_tet;
        double acc = 0.0;
        for (int t = 0; t < n_tet; ++t) acc += row[t] * R[t];
        v_sph[uv] += acc;
      }
    }
  }
}

#if defined(GPU4DFT_HAVE_CUDA)
int snk_cuda_device_count();
void snk_esp_matrix_cuda(const detail::FlatBasisData& flat_basis,
                         const std::vector<SnkShellPair>& pairs,
                         double gx, double gy, double gz,
                         double pair_cutoff,
                         double* out_v);
void* snk_cuda_create(const detail::FlatBasisData& flat_basis,
                      const MolecularGrid& grid,
                      const SnkPairTable& pair_table,
                      const std::vector<double>& s_exact,
                      const SnkOptions& opts,
                      std::vector<double>& q_fit_out);
void snk_cuda_destroy(void* impl) noexcept;
void snk_cuda_compute_k_device(void* impl,
                               const double* d_dm,
                               double* d_vk,
                               void* cuda_stream);
void snk_cuda_compute_k_host(void* impl,
                             const double* h_dm,
                             double* h_vk);
#endif

}  // namespace snk_detail

struct SemiNumericalK::HostPairState {
  detail::FlatBasisData flat_basis;
  snk_detail::SnkPairTable pair_table;

  HostPairState(const Basis& b, const std::vector<double>& s_exact, double cutoff)
      : flat_basis(b),
        pair_table(snk_detail::build_snk_pair_table(b, flat_basis, s_exact, cutoff)) {}
};

#if defined(GPU4DFT_HAVE_CUDA)
struct SemiNumericalK::CudaState {
  void* impl = nullptr;
};
#else
struct SemiNumericalK::CudaState {};
#endif

namespace {

void check_supported_l(const Basis& basis, const char* caller) {
  if (basis.max_l() > 4) {
    throw NotImplemented(std::string(caller) +
                         " supports up to g shells (l <= 4); basis has l = " +
                         std::to_string(basis.max_l()));
  }
}

void host_gemm_nn(int M, int N, int K, const double* A, const double* B, double* C,
                  bool accum = false) {
  if (!accum) std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0);
  for (int i = 0; i < M; ++i) {
    double* ci = C + static_cast<std::size_t>(i) * N;
    const double* ai = A + static_cast<std::size_t>(i) * K;
    for (int k = 0; k < K; ++k) {
      const double aik = ai[k];
      if (aik == 0.0) continue;
      const double* bk = B + static_cast<std::size_t>(k) * N;
      for (int j = 0; j < N; ++j) ci[j] += aik * bk[j];
    }
  }
}

void host_gemm_nt(int M, int N, int K, const double* A, const double* B, double* C,
                  bool accum = false) {
  if (!accum) std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0);
  for (int i = 0; i < M; ++i) {
    const double* ai = A + static_cast<std::size_t>(i) * K;
    double* ci = C + static_cast<std::size_t>(i) * N;
    for (int j = 0; j < N; ++j) {
      const double* bj = B + static_cast<std::size_t>(j) * K;
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += ai[k] * bj[k];
      ci[j] += s;
    }
  }
}

void host_gemm_tn(int M, int N, int K, const double* A, const double* B, double* C,
                  bool accum = false) {
  if (!accum) std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0);
  for (int k = 0; k < K; ++k) {
    const double* ak = A + static_cast<std::size_t>(k) * M;
    const double* bk = B + static_cast<std::size_t>(k) * N;
    for (int i = 0; i < M; ++i) {
      const double aki = ak[i];
      if (aki == 0.0) continue;
      double* ci = C + static_cast<std::size_t>(i) * N;
      for (int j = 0; j < N; ++j) ci[j] += aki * bk[j];
    }
  }
}

// Cyclic Jacobi symmetric eigensolver for A [n x n] -> eigenvalues w [n], eigenvectors V [n x n] (columns)
void jacobi_syev(int n, std::vector<double> A, std::vector<double>& w, std::vector<double>& V) {
  V.assign(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) V[static_cast<std::size_t>(i) * n + i] = 1.0;

  for (int sweep = 0; sweep < 80; ++sweep) {
    double off_norm2 = 0.0;
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = A[static_cast<std::size_t>(p) * n + q];
        off_norm2 += apq * apq;
      }
    }
    if (std::sqrt(off_norm2) < 1e-15 * static_cast<double>(n)) break;

    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = A[static_cast<std::size_t>(p) * n + q];
        if (std::abs(apq) < 1e-18) continue;
        const double app = A[static_cast<std::size_t>(p) * n + p];
        const double aqq = A[static_cast<std::size_t>(q) * n + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0)
                             ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                             : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        A[static_cast<std::size_t>(p) * n + p] = app - t * apq;
        A[static_cast<std::size_t>(q) * n + q] = aqq + t * apq;
        A[static_cast<std::size_t>(p) * n + q] = 0.0;
        A[static_cast<std::size_t>(q) * n + p] = 0.0;

        for (int r = 0; r < n; ++r) {
          if (r == p || r == q) continue;
          const double arp = A[static_cast<std::size_t>(r) * n + p];
          const double arq = A[static_cast<std::size_t>(r) * n + q];
          A[static_cast<std::size_t>(r) * n + p] = c * arp - s * arq;
          A[static_cast<std::size_t>(p) * n + r] = A[static_cast<std::size_t>(r) * n + p];
          A[static_cast<std::size_t>(r) * n + q] = s * arp + c * arq;
          A[static_cast<std::size_t>(q) * n + r] = A[static_cast<std::size_t>(r) * n + q];
        }

        for (int r = 0; r < n; ++r) {
          const double vrp = V[static_cast<std::size_t>(r) * n + p];
          const double vrq = V[static_cast<std::size_t>(r) * n + q];
          V[static_cast<std::size_t>(r) * n + p] = c * vrp - s * vrq;
          V[static_cast<std::size_t>(r) * n + q] = s * vrp + c * vrq;
        }
      }
    }
  }

  w.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = A[static_cast<std::size_t>(i) * n + i];
}

std::vector<double> compute_cosx_q_host(int nao,
                                        const std::vector<double>& s_grid,
                                        const std::vector<double>& s_exact,
                                        double lindep_tol) {
  const std::size_t n = static_cast<std::size_t>(nao);
  std::vector<double> q_identity(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) q_identity[i * n + i] = 1.0;
  if (nao == 0) return q_identity;

  std::vector<double> evals;
  std::vector<double> evecs;
  jacobi_syev(nao, s_grid, evals, evecs);

  std::vector<double> sg_pinv(n * n, 0.0);
  std::vector<double> p_null(n * n, 0.0);
  for (std::size_t k = 0; k < n; ++k) {
    const double lam = evals[k];
    const bool keep = (lam > lindep_tol);
    const double inv_lam = keep ? (1.0 / lam) : 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double vik = evecs[i * n + k];
      for (std::size_t j = 0; j < n; ++j) {
        const double vjk = evecs[j * n + k];
        if (keep) sg_pinv[i * n + j] += inv_lam * vik * vjk;
        else p_null[i * n + j] += vik * vjk;
      }
    }
  }

  std::vector<double> q(n * n, 0.0);
  host_gemm_nn(nao, nao, nao, sg_pinv.data(), s_exact.data(), q.data(), false);
  for (std::size_t idx = 0; idx < n * n; ++idx) {
    q[idx] += p_null[idx];
  }
  return q;
}

std::vector<double> compute_grid_overlap_host(const Basis& basis, const MolecularGrid& grid) {
  const std::size_t nao = static_cast<std::size_t>(basis.nao());
  std::vector<double> s_grid(nao * nao, 0.0);
  if (nao == 0 || grid.npoints() == 0) return s_grid;

  const detail::FlatBasisData flat(basis);
  const detail::FlatBasisView bview = flat.view();
  const auto& batches = grid.batches();
  const auto& coords = grid.coords();
  const auto& weights = grid.weights();

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> local_sg(nao * nao, 0.0);
    std::vector<double> phi;
    std::vector<double> phi_w;
    std::vector<double> sg_act;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const SpatialBatch& b = batches[static_cast<std::size_t>(ib)];
      const std::size_t ng = b.npoints;
      const std::size_t na = b.active_aos.size();
      if (ng == 0 || na == 0) continue;

      phi.assign(na * ng, 0.0);
      std::size_t cur = 0;
      for (int is : b.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = b.point_offset + g;
          detail::eval_shell_at_point(
              bview, is,
              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
              0, phi.data() + cur * ng + g, na * ng, ng);
        }
        cur += static_cast<std::size_t>(nm);
      }

      phi_w.resize(na * ng);
      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          phi_w[u * ng + g] = weights[b.point_offset + g] * phi[u * ng + g];
        }
      }

      sg_act.resize(na * na);
      host_gemm_nt(static_cast<int>(na), static_cast<int>(na), static_cast<int>(ng),
                   phi.data(), phi_w.data(), sg_act.data(), false);

      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(b.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(b.active_aos[v]);
          local_sg[gu * nao + gv] += sg_act[u * na + v];
        }
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t k = 0; k < nao * nao; ++k) s_grid[k] += local_sg[k];
    }
  }
  return s_grid;
}

}  // namespace

std::vector<double> snk_overlap_matrix(const Basis& basis) {
  check_supported_l(basis, "snk_overlap_matrix");
  constexpr double kPi = 3.141592653589793238462643383279502884;
  const std::size_t nao = static_cast<std::size_t>(basis.nao());
  std::vector<double> S(nao * nao, 0.0);
  if (nao == 0) return S;

  const detail::FlatBasisData flat(basis);
  const detail::FlatBasisView b = flat.view();

  for (int ia = 0; ia < b.nshell; ++ia) {
    const int la = b.shell_l[ia];
    const int nma = 2 * la + 1;
    const int ao_a = b.shell_ao_offset[ia];
    snk_detail::SphCartTerm ta[28];
    const int nta = snk_detail::sph_cart_terms(la, ta);

    for (int ib = 0; ib <= ia; ++ib) {
      const int lb = b.shell_l[ib];
      const int nmb = 2 * lb + 1;
      const int ao_b = b.shell_ao_offset[ib];
      snk_detail::SphCartTerm tb[28];
      const int ntb = snk_detail::sph_cart_terms(lb, tb);

      const double ax = b.shell_center[3 * ia + 0], ay = b.shell_center[3 * ia + 1], az = b.shell_center[3 * ia + 2];
      const double bx = b.shell_center[3 * ib + 0], by = b.shell_center[3 * ib + 1], bz = b.shell_center[3 * ib + 2];
      const double rab2 = (ax - bx) * (ax - bx) + (ay - by) * (ay - by) + (az - bz) * (az - bz);

      double block[81] = {0.0};
      double Ex[5][5][9], Ey[5][5][9], Ez[5][5][9];

      const int pa0 = b.shell_prim_offset[ia], npa = b.shell_prim_count[ia];
      const int pb0 = b.shell_prim_offset[ib], npb = b.shell_prim_count[ib];

      for (int ka = 0; ka < npa; ++ka) {
        const double alpha = b.prim_exp[pa0 + ka];
        const double ca = b.prim_coeff[pa0 + ka];
        for (int kb = 0; kb < npb; ++kb) {
          const double beta = b.prim_exp[pb0 + kb];
          const double cb = b.prim_coeff[pb0 + kb];
          const double p = alpha + beta;
          const double inv_p = 1.0 / p;
          const double mu = alpha * beta * inv_p;
          if (mu * rab2 > 45.0) continue;

          const double px = (alpha * ax + beta * bx) * inv_p;
          const double py = (alpha * ay + beta * by) * inv_p;
          const double pz = (alpha * az + beta * bz) * inv_p;

          snk_detail::md_hermite_1d(la, lb, px - ax, px - bx, 0.5 * inv_p, Ex);
          snk_detail::md_hermite_1d(la, lb, py - ay, py - by, 0.5 * inv_p, Ey);
          snk_detail::md_hermite_1d(la, lb, pz - az, pz - bz, 0.5 * inv_p, Ez);

          const double s00 = std::pow(kPi * inv_p, 1.5) * std::exp(-mu * rab2) * ca * cb;
          for (int t_i = 0; t_i < nta; ++t_i) {
            const auto& sa = ta[t_i];
            for (int t_j = 0; t_j < ntb; ++t_j) {
              const auto& sb = tb[t_j];
              const double cart_ovlp =
                  Ex[sa.ix][sb.ix][0] * Ey[sa.iy][sb.iy][0] * Ez[sa.iz][sb.iz][0];
              block[sa.m * nmb + sb.m] += s00 * (sa.coeff * sb.coeff) * cart_ovlp;
            }
          }
        }
      }

      for (int ma = 0; ma < nma; ++ma) {
        for (int mb = 0; mb < nmb; ++mb) {
          const double val = block[ma * nmb + mb];
          const std::size_t u = static_cast<std::size_t>(ao_a + ma);
          const std::size_t v = static_cast<std::size_t>(ao_b + mb);
          S[u * nao + v] = val;
          S[v * nao + u] = val;
        }
      }
    }
  }
  return S;
}

std::vector<double> snk_esp_matrix(const Basis& basis,
                                   std::array<double, 3> point_bohr,
                                   Backend backend) {
  check_supported_l(basis, "snk_esp_matrix");
  const std::size_t nao = static_cast<std::size_t>(basis.nao());
  std::vector<double> V(nao * nao, 0.0);
  if (nao == 0) return V;

  const detail::FlatBasisData flat(basis);
  const std::vector<double> empty_s;
  const auto tbl = snk_detail::build_snk_pair_table(basis, flat, empty_s, 1e-16);

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (snk_detail::snk_cuda_device_count() == 0) {
      throw std::runtime_error("snk_esp_matrix: no CUDA device available");
    }
    snk_detail::snk_esp_matrix_cuda(flat, tbl.pairs,
                                    point_bohr[0], point_bohr[1], point_bohr[2],
                                    1e-16, V.data());
    return V;
#else
    throw std::runtime_error("snk_esp_matrix: gpu4dft was built without CUDA");
#endif
  }
  if (backend != Backend::Host) {
    throw std::runtime_error("snk_esp_matrix: unsupported backend");
  }

  const detail::FlatBasisView bview = flat.view();
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
  for (std::ptrdiff_t ip = 0; ip < static_cast<std::ptrdiff_t>(tbl.pairs.size()); ++ip) {
    const auto& pr = tbl.pairs[static_cast<std::size_t>(ip)];
    const int ia = pr.ish_a;
    const int ib = pr.ish_b;
    const int nma = pr.nma;
    const int nmb = pr.nmb;
    const int ao_a = pr.ao_a;
    const int ao_b = pr.ao_b;

    double block[81];
    const int L = pr.la + pr.lb;
    if (L == 0) {
      snk_detail::eval_precomputed_pair_esp_host<0>(
          pr, tbl.prims.data(), tbl.h_coeffs.data(),
          point_bohr[0], point_bohr[1], point_bohr[2], block);
    } else if (L == 1) {
      snk_detail::eval_precomputed_pair_esp_host<1>(
          pr, tbl.prims.data(), tbl.h_coeffs.data(),
          point_bohr[0], point_bohr[1], point_bohr[2], block);
    } else if (L == 2) {
      snk_detail::eval_precomputed_pair_esp_host<2>(
          pr, tbl.prims.data(), tbl.h_coeffs.data(),
          point_bohr[0], point_bohr[1], point_bohr[2], block);
    } else if (L == 3) {
      snk_detail::eval_precomputed_pair_esp_host<3>(
          pr, tbl.prims.data(), tbl.h_coeffs.data(),
          point_bohr[0], point_bohr[1], point_bohr[2], block);
    } else if (L == 4) {
      snk_detail::eval_precomputed_pair_esp_host<4>(
          pr, tbl.prims.data(), tbl.h_coeffs.data(),
          point_bohr[0], point_bohr[1], point_bohr[2], block);
    } else {
      snk_detail::eval_shell_pair_esp(bview, ia, ib,
                                      point_bohr[0], point_bohr[1], point_bohr[2],
                                      1e-16, block);
    }

    for (int ma = 0; ma < nma; ++ma) {
      for (int mb = 0; mb < nmb; ++mb) {
        const double val = block[ma * nmb + mb];
        const std::size_t u = static_cast<std::size_t>(ao_a + ma);
        const std::size_t v = static_cast<std::size_t>(ao_b + mb);
        V[u * nao + v] = val;
        V[v * nao + u] = val;
      }
    }
  }
  return V;
}

SemiNumericalK::SemiNumericalK(const Basis& basis, const MolecularGrid& grid,
                               const SnkOptions& opts)
    : basis_(basis), grid_(grid), opts_(opts) {
  check_supported_l(basis_, "SemiNumericalK");
  if (grid_.nao() != basis_.nao() && grid_.npoints() > 0) {
    throw std::invalid_argument("SemiNumericalK: grid.nao() does not match basis.nao()");
  }

  s_exact_ = snk_overlap_matrix(basis_);
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  q_fit_.assign(nao * nao, 0.0);
  for (std::size_t i = 0; i < nao; ++i) q_fit_[i * nao + i] = 1.0;

  if (opts_.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (snk_detail::snk_cuda_device_count() > 0) {
      const detail::FlatBasisData flat(basis_);
      const auto pair_table =
          snk_detail::build_snk_pair_table(basis_, flat, s_exact_, opts_.pair_cutoff);
      cuda_ = std::make_unique<CudaState>();
      cuda_->impl =
          snk_detail::snk_cuda_create(flat, grid_, pair_table, s_exact_, opts_, q_fit_);
      active_backend_ = Backend::Cuda;
      return;
    }
    throw std::runtime_error("SemiNumericalK: Backend::Cuda requested, but no CUDA device is available");
#else
    throw std::runtime_error("SemiNumericalK: gpu4dft was built without CUDA");
#endif
  }
  if (opts_.backend != Backend::Host) {
    throw std::runtime_error("SemiNumericalK: unsupported backend");
  }

  active_backend_ = Backend::Host;
  host_pairs_ = std::make_unique<HostPairState>(basis_, s_exact_, opts_.pair_cutoff);
  if (opts_.overlap_fitting && nao > 0 && grid_.npoints() > 0) {
    const std::vector<double> s_grid = compute_grid_overlap_host(basis_, grid_);
    q_fit_ = compute_cosx_q_host(basis_.nao(), s_grid, s_exact_, opts_.lindep_tol);
  }
}

SemiNumericalK::~SemiNumericalK() {
#if defined(GPU4DFT_HAVE_CUDA)
  if (cuda_ && cuda_->impl) {
    snk_detail::snk_cuda_destroy(cuda_->impl);
    cuda_->impl = nullptr;
  }
#endif
}

SemiNumericalK::SemiNumericalK(SemiNumericalK&&) noexcept = default;
SemiNumericalK& SemiNumericalK::operator=(SemiNumericalK&&) noexcept = default;

std::vector<double> SemiNumericalK::compute_k(std::span<const double> dm) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  if (dm.size() != nao * nao) {
    throw std::invalid_argument("SemiNumericalK::compute_k: dm size must equal nao * nao");
  }
  std::vector<double> vk(nao * nao, 0.0);
  compute_k(dm.data(), vk.data());
  return vk;
}

void SemiNumericalK::compute_k(const double* dm, double* vk_out) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  std::fill_n(vk_out, nao * nao, 0.0);
  if (nao == 0 || grid_.npoints() == 0) return;

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    snk_detail::snk_cuda_compute_k_host(cuda_->impl, dm, vk_out);
    return;
#endif
  }

  const detail::FlatBasisView bview = host_pairs_->flat_basis.view();
  const auto& tbl = host_pairs_->pair_table;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();
  const int i_nao = static_cast<int>(nao);
  const int nshell = bview.nshell;

  std::vector<double> dm_sym(nao * nao);
  for (std::size_t i = 0; i < nao; ++i) {
    for (std::size_t j = 0; j < nao; ++j) {
      dm_sym[i * nao + j] = 0.5 * (dm[i * nao + j] + dm[j * nao + i]);
    }
  }

  std::vector<double> k_raw(nao * nao, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> local_k(nao * nao, 0.0);
    std::vector<double> phi;
    std::vector<double> phi_w;
    std::vector<double> d_cols;
    std::vector<double> f_mat;
    std::vector<double> g_mat;
    std::vector<double> k_batch;
    std::vector<double> f_sh_max(static_cast<std::size_t>(nshell), 0.0);

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const SpatialBatch& b = batches[static_cast<std::size_t>(ib)];
      const std::size_t ng = b.npoints;
      const std::size_t na = b.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);

      phi.assign(na * ng, 0.0);
      std::size_t cur = 0;
      for (int is : b.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = b.point_offset + g;
          detail::eval_shell_at_point(
              bview, is,
              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
              0, phi.data() + cur * ng + g, na * ng, ng);
        }
        cur += static_cast<std::size_t>(nm);
      }

      d_cols.resize(nao * na);
      for (std::size_t v = 0; v < nao; ++v) {
        for (std::size_t u = 0; u < na; ++u) {
          d_cols[v * na + u] = dm_sym[v * nao + static_cast<std::size_t>(b.active_aos[u])];
        }
      }

      f_mat.resize(nao * ng);
      host_gemm_nn(i_nao, i_ng, i_na, d_cols.data(), phi.data(), f_mat.data(), false);

      // Compute per-shell max |F| across this spatial batch for exact Cauchy-Schwarz screening
      for (int is = 0; is < nshell; ++is) {
        const int nm = 2 * bview.shell_l[is] + 1;
        const std::size_t ao0 = static_cast<std::size_t>(bview.shell_ao_offset[is]);
        double mval = 0.0;
        for (int m = 0; m < nm; ++m) {
          const double* row = f_mat.data() + (ao0 + static_cast<std::size_t>(m)) * ng;
          for (std::size_t g = 0; g < ng; ++g) {
            mval = std::max(mval, std::abs(row[g]));
          }
        }
        f_sh_max[static_cast<std::size_t>(is)] = mval;
      }

      g_mat.assign(nao * ng, 0.0);
      for (const auto& pr : tbl.pairs) {
        const int sa = pr.ish_a;
        const int sb = pr.ish_b;
        const double f_ub = std::max(f_sh_max[static_cast<std::size_t>(sa)],
                                     f_sh_max[static_cast<std::size_t>(sb)]);
        if (f_ub == 0.0) continue;

        double d2_box = 0.0;
        for (int d = 0; d < 3; ++d) {
          double dd = 0.0;
          if (pr.center[d] < b.box_min[d]) dd = b.box_min[d] - pr.center[d];
          else if (pr.center[d] > b.box_max[d]) dd = pr.center[d] - b.box_max[d];
          d2_box += dd * dd;
        }
        const double dist_box = std::sqrt(d2_box);
        const double v_ub = (dist_box > pr.extent_radius + 1.0)
                                ? (pr.q_far_bound / (dist_box - pr.extent_radius))
                                : pr.v_max_near;
        if (v_ub * f_ub < 1e-14) continue;

        const int nma = pr.nma;
        const int nmb = pr.nmb;
        const std::size_t ao_a = static_cast<std::size_t>(pr.ao_a);
        const std::size_t ao_b = static_cast<std::size_t>(pr.ao_b);
        const int L = pr.la + pr.lb;

        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = b.point_offset + g;
          const double gx = coords[3 * gp + 0];
          const double gy = coords[3 * gp + 1];
          const double gz = coords[3 * gp + 2];

          double v_block[81];
          if (L == 0) {
            snk_detail::eval_precomputed_pair_esp_host<0>(
                pr, tbl.prims.data(), tbl.h_coeffs.data(), gx, gy, gz, v_block);
          } else if (L == 1) {
            snk_detail::eval_precomputed_pair_esp_host<1>(
                pr, tbl.prims.data(), tbl.h_coeffs.data(), gx, gy, gz, v_block);
          } else if (L == 2) {
            snk_detail::eval_precomputed_pair_esp_host<2>(
                pr, tbl.prims.data(), tbl.h_coeffs.data(), gx, gy, gz, v_block);
          } else if (L == 3) {
            snk_detail::eval_precomputed_pair_esp_host<3>(
                pr, tbl.prims.data(), tbl.h_coeffs.data(), gx, gy, gz, v_block);
          } else if (L == 4) {
            snk_detail::eval_precomputed_pair_esp_host<4>(
                pr, tbl.prims.data(), tbl.h_coeffs.data(), gx, gy, gz, v_block);
          } else {
            snk_detail::eval_shell_pair_esp(
                bview, sa, sb, gx, gy, gz, opts_.pair_cutoff, v_block);
          }

          for (int ma = 0; ma < nma; ++ma) {
            const std::size_t ua = ao_a + static_cast<std::size_t>(ma);
            const double fa_g = f_mat[ua * ng + g];
            double acc_a = 0.0;
            for (int mb = 0; mb < nmb; ++mb) {
              const std::size_t ub = ao_b + static_cast<std::size_t>(mb);
              const double a_uv = v_block[ma * nmb + mb];
              acc_a += a_uv * f_mat[ub * ng + g];
              if (sa != sb) {
                g_mat[ub * ng + g] += a_uv * fa_g;
              }
            }
            g_mat[ua * ng + g] += acc_a;
          }
        }
      }

      phi_w.resize(na * ng);
      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          phi_w[u * ng + g] = weights[b.point_offset + g] * phi[u * ng + g];
        }
      }

      k_batch.resize(na * nao);
      host_gemm_nt(i_na, i_nao, i_ng, phi_w.data(), g_mat.data(), k_batch.data(), false);

      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(b.active_aos[u]);
        for (std::size_t v = 0; v < nao; ++v) {
          local_k[gu * nao + v] += k_batch[u * nao + v];
        }
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t k = 0; k < nao * nao; ++k) k_raw[k] += local_k[k];
    }
  }

  if (opts_.overlap_fitting) {
    std::vector<double> qtk(nao * nao, 0.0);
    host_gemm_tn(i_nao, i_nao, i_nao, q_fit_.data(), k_raw.data(), qtk.data(), false);
    for (std::size_t i = 0; i < nao; ++i) {
      for (std::size_t j = 0; j < nao; ++j) {
        vk_out[i * nao + j] = 0.5 * (qtk[i * nao + j] + qtk[j * nao + i]);
      }
    }
  } else {
    for (std::size_t i = 0; i < nao; ++i) {
      for (std::size_t j = 0; j < nao; ++j) {
        vk_out[i * nao + j] = 0.5 * (k_raw[i * nao + j] + k_raw[j * nao + i]);
      }
    }
  }
}

void SemiNumericalK::compute_k_device(const double* d_dm, double* d_vk, void* cuda_stream) const {
  if (active_backend_ != Backend::Cuda) {
    throw std::runtime_error("SemiNumericalK::compute_k_device requires Backend::Cuda");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  snk_detail::snk_cuda_compute_k_device(cuda_->impl, d_dm, d_vk, cuda_stream);
#else
  (void)d_dm;
  (void)d_vk;
  (void)cuda_stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

}  // namespace gpu4dft

// -----------------------------------------------------------------------------
// C-ABI entry points for standalone & integrated Python ctypes verification
// -----------------------------------------------------------------------------
#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace {

gpu4dft::Basis build_basis_from_c(int nshell,
                                  const int* sh_l,
                                  const double* sh_center,
                                  const int* sh_nprim,
                                  const double* prim_exp,
                                  const double* prim_coeff) {
  std::vector<gpu4dft::Shell> shells;
  shells.reserve(static_cast<std::size_t>(nshell));
  int p_off = 0;
  for (int i = 0; i < nshell; ++i) {
    const int np = sh_nprim[i];
    gpu4dft::Shell sh;
    sh.l = sh_l[i];
    sh.center = {sh_center[3 * i + 0], sh_center[3 * i + 1], sh_center[3 * i + 2]};
    sh.exponents.assign(prim_exp + p_off, prim_exp + p_off + np);
    sh.coefficients.assign(prim_coeff + p_off, prim_coeff + p_off + np);
    p_off += np;
    shells.push_back(std::move(sh));
  }
  return gpu4dft::Basis(std::move(shells));
}

std::vector<gpu4dft::XcAtom> build_atoms_from_c(int natom,
                                                const int* atomic_nums,
                                                const double* atom_coords) {
  std::vector<gpu4dft::XcAtom> atoms(static_cast<std::size_t>(natom));
  for (int i = 0; i < natom; ++i) {
    atoms[static_cast<std::size_t>(i)] = gpu4dft::XcAtom{
        atomic_nums[i],
        {atom_coords[3 * i + 0], atom_coords[3 * i + 1], atom_coords[3 * i + 2]}};
  }
  return atoms;
}

}  // namespace

extern "C" {

int gpu4dft_xc_c_cuda_available() {
#if defined(GPU4DFT_HAVE_CUDA)
  return gpu4dft::snk_detail::snk_cuda_device_count() > 0 ? 1 : 0;
#else
  return 0;
#endif
}

double gpu4dft_xc_c_exx_fraction(const char* xc_name) {
  return gpu4dft::xc_exact_exchange_fraction(xc_name);
}

int gpu4dft_xc_c_lebedev(int n_ang, double* coords_out, double* weights_out) {
  try {
    auto [xyz, w] = gpu4dft::MolecularGrid::lebedev_grid(n_ang);
    std::copy(xyz.begin(), xyz.end(), coords_out);
    std::copy(w.begin(), w.end(), weights_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_eval_xc(const char* xc_name,
                         int n,
                         const double* rho,
                         const double* sigma,
                         const double* tau,
                         int deriv,
                         int backend,
                         double cutoff,
                         double* out_10n) {
  try {
    const auto bk = (backend == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    const std::size_t un = static_cast<std::size_t>(n);
    const auto res = gpu4dft::eval_xc_points(
        xc_name,
        std::span<const double>(rho, un),
        sigma ? std::span<const double>(sigma, un) : std::span<const double>{},
        tau ? std::span<const double>(tau, un) : std::span<const double>{},
        deriv, bk, cutoff);
    std::fill_n(out_10n, 10 * un, 0.0);
    std::copy(res.zk.begin(), res.zk.end(), out_10n + 0 * un);
    if (deriv >= 1) {
      std::copy(res.vrho.begin(), res.vrho.end(), out_10n + 1 * un);
      std::copy(res.vsigma.begin(), res.vsigma.end(), out_10n + 2 * un);
      std::copy(res.vtau.begin(), res.vtau.end(), out_10n + 3 * un);
    }
    if (deriv >= 2) {
      std::copy(res.v2rho2.begin(), res.v2rho2.end(), out_10n + 4 * un);
      std::copy(res.v2rhosigma.begin(), res.v2rhosigma.end(), out_10n + 5 * un);
      std::copy(res.v2sigma2.begin(), res.v2sigma2.end(), out_10n + 6 * un);
      std::copy(res.v2rhotau.begin(), res.v2rhotau.end(), out_10n + 7 * un);
      std::copy(res.v2sigmatau.begin(), res.v2sigmatau.end(), out_10n + 8 * un);
      std::copy(res.v2tau2.begin(), res.v2tau2.end(), out_10n + 9 * un);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_eval_ao(int nshell,
                         const int* sh_l,
                         const double* sh_center,
                         const int* sh_nprim,
                         const double* prim_exp,
                         const double* prim_coeff,
                         int npts,
                         const double* coords,
                         int deriv,
                         int backend,
                         double* out_ao) {
  try {
    const auto basis = build_basis_from_c(nshell, sh_l, sh_center, sh_nprim, prim_exp, prim_coeff);
    const auto bk = (backend == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    const auto ao = gpu4dft::eval_ao_grid(
        basis, std::span<const double>(coords, static_cast<std::size_t>(npts) * 3), deriv, bk);
    std::copy(ao.begin(), ao.end(), out_ao);
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_build_grid(int natom,
                            const int* atomic_nums,
                            const double* atom_coords,
                            int nshell,
                            const int* sh_l,
                            const double* sh_center,
                            const int* sh_nprim,
                            const double* prim_exp,
                            const double* prim_coeff,
                            int level,
                            int partition_scheme,
                            int backend,
                            int max_out_pts,
                            int* out_npts,
                            int* out_nbatches,
                            double* out_avg_active_aos,
                            double* coords_out,
                            double* weights_out) {
  try {
    const auto basis = build_basis_from_c(nshell, sh_l, sh_center, sh_nprim, prim_exp, prim_coeff);
    const auto atoms = build_atoms_from_c(natom, atomic_nums, atom_coords);
    gpu4dft::GridOptions opts;
    opts.level = level;
    opts.partition = (partition_scheme == 1) ? gpu4dft::PartitionScheme::Becke
                                             : gpu4dft::PartitionScheme::Stratmann;
    opts.backend = (backend == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    gpu4dft::MolecularGrid grid(atoms, basis, opts);

    *out_npts = static_cast<int>(grid.npoints());
    *out_nbatches = static_cast<int>(grid.nbatches());
    double sum_act = 0.0;
    for (const auto& b : grid.batches()) sum_act += static_cast<double>(b.active_aos.size());
    *out_avg_active_aos = grid.batches().empty()
                              ? 0.0
                              : (sum_act / static_cast<double>(grid.batches().size()));
    if (coords_out && weights_out && static_cast<int>(grid.npoints()) <= max_out_pts) {
      std::copy(grid.coords().begin(), grid.coords().end(), coords_out);
      std::copy(grid.weights().begin(), grid.weights().end(), weights_out);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_eval_vxc(int nshell,
                          const int* sh_l,
                          const double* sh_center,
                          const int* sh_nprim,
                          const double* prim_exp,
                          const double* prim_coeff,
                          int npts,
                          const double* grid_coords,
                          const double* grid_weights,
                          const char* xc_name,
                          int backend,
                          int precision,
                          int use_device_api,
                          const double* dm,
                          double* exc_out,
                          double* nelec_out,
                          double* vxc_out) {
  try {
    const auto basis = build_basis_from_c(nshell, sh_l, sh_center, sh_nprim, prim_exp, prim_coeff);
    const std::size_t un = static_cast<std::size_t>(npts);
    const auto grid = gpu4dft::MolecularGrid::from_points(
        std::span<const double>(grid_coords, un * 3),
        std::span<const double>(grid_weights, un),
        basis, 2048, 1e-16);
    gpu4dft::XcOptions opts;
    opts.backend = (backend == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    opts.precision = (precision == 1) ? gpu4dft::Precision::FP32 : gpu4dft::Precision::FP64;
    gpu4dft::XcEngine eng(basis, grid, xc_name, opts);

    if (use_device_api && opts.backend == gpu4dft::Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
      const std::size_t nao2 = static_cast<std::size_t>(basis.nao()) * basis.nao();
      double *d_dm = nullptr, *d_vxc = nullptr;
      if (cudaMalloc(&d_dm, nao2 * sizeof(double)) != cudaSuccess) return -2;
      if (cudaMalloc(&d_vxc, nao2 * sizeof(double)) != cudaSuccess) {
        cudaFree(d_dm);
        return -2;
      }
      cudaMemcpy(d_dm, dm, nao2 * sizeof(double), cudaMemcpyHostToDevice);
      *exc_out = eng.eval_vxc_device(d_dm, d_vxc, nullptr);
      if (nelec_out) *nelec_out = 0.0;
      cudaMemcpy(vxc_out, d_vxc, nao2 * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(d_dm);
      cudaFree(d_vxc);
      return 0;
#else
      return -1;
#endif
    }

    *exc_out = eng.eval_vxc(dm, vxc_out, nelec_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_eval_fxc(int nshell,
                          const int* sh_l,
                          const double* sh_center,
                          const int* sh_nprim,
                          const double* prim_exp,
                          const double* prim_coeff,
                          int npts,
                          const double* grid_coords,
                          const double* grid_weights,
                          const char* xc_name,
                          int backend,
                          int precision,
                          int use_device_api,
                          const double* dm0,
                          int n_pert,
                          const double* delta_dms,
                          double* fxc_out) {
  try {
    const auto basis = build_basis_from_c(nshell, sh_l, sh_center, sh_nprim, prim_exp, prim_coeff);
    const std::size_t un = static_cast<std::size_t>(npts);
    const auto grid = gpu4dft::MolecularGrid::from_points(
        std::span<const double>(grid_coords, un * 3),
        std::span<const double>(grid_weights, un),
        basis, 2048, 1e-16);
    gpu4dft::XcOptions opts;
    opts.backend = (backend == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    opts.precision = (precision == 1) ? gpu4dft::Precision::FP32 : gpu4dft::Precision::FP64;
    gpu4dft::XcEngine eng(basis, grid, xc_name, opts);

    const std::size_t nao2 = static_cast<std::size_t>(basis.nao()) * basis.nao();

    if (use_device_api && opts.backend == gpu4dft::Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
      double *d_dm0 = nullptr, *d_ddm = nullptr, *d_fxc = nullptr;
      if (cudaMalloc(&d_dm0, nao2 * sizeof(double)) != cudaSuccess) return -2;
      if (cudaMalloc(&d_ddm, nao2 * sizeof(double)) != cudaSuccess) {
        cudaFree(d_dm0);
        return -2;
      }
      if (cudaMalloc(&d_fxc, nao2 * sizeof(double)) != cudaSuccess) {
        cudaFree(d_dm0);
        cudaFree(d_ddm);
        return -2;
      }
      cudaMemcpy(d_dm0, dm0, nao2 * sizeof(double), cudaMemcpyHostToDevice);
      eng.set_reference_density_device(d_dm0, nullptr);
      for (int p = 0; p < n_pert; ++p) {
        cudaMemcpy(d_ddm, delta_dms + static_cast<std::size_t>(p) * nao2,
                   nao2 * sizeof(double), cudaMemcpyHostToDevice);
        eng.eval_fxc_device(d_ddm, d_fxc, nullptr);
        cudaMemcpy(fxc_out + static_cast<std::size_t>(p) * nao2, d_fxc,
                   nao2 * sizeof(double), cudaMemcpyDeviceToHost);
      }
      cudaFree(d_dm0);
      cudaFree(d_ddm);
      cudaFree(d_fxc);
      return 0;
#else
      return -1;
#endif
    }

    eng.set_reference_density(dm0);
    for (int p = 0; p < n_pert; ++p) {
      eng.eval_fxc(delta_dms + static_cast<std::size_t>(p) * nao2,
                   fxc_out + static_cast<std::size_t>(p) * nao2);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_nuclear_grad(int natom,
                              const int* atomic_nums,
                              const double* atom_coords,
                              int nshell,
                              const int* sh_l,
                              const double* sh_center,
                              const int* sh_nprim,
                              const double* prim_exp,
                              const double* prim_coeff,
                              int npts,
                              const double* grid_coords,
                              const double* grid_weights,
                              const char* xc_name,
                              int backend,
                              const double* dm,
                              double* grad_out) {
  try {
    const auto basis = build_basis_from_c(nshell, sh_l, sh_center, sh_nprim, prim_exp, prim_coeff);
    const auto atoms = build_atoms_from_c(natom, atomic_nums, atom_coords);
    const std::size_t un = static_cast<std::size_t>(npts);
    const auto grid = gpu4dft::MolecularGrid::from_points(
        std::span<const double>(grid_coords, un * 3),
        std::span<const double>(grid_weights, un),
        basis, 2048, 1e-16);
    const auto bk = (backend == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    const auto g = gpu4dft::xc_nuclear_grad(basis, grid, atoms, xc_name, dm, bk);
    std::copy(g.begin(), g.end(), grad_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_snk_overlap(int nshell,
                             const int* sh_l,
                             const double* sh_center,
                             const int* sh_nprim,
                             const double* prim_exp,
                             const double* prim_coeff,
                             double* s_out) {
  try {
    const auto basis = build_basis_from_c(nshell, sh_l, sh_center, sh_nprim, prim_exp, prim_coeff);
    const auto s = gpu4dft::snk_overlap_matrix(basis);
    std::copy(s.begin(), s.end(), s_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_snk_esp(int nshell,
                         const int* sh_l,
                         const double* sh_center,
                         const int* sh_nprim,
                         const double* prim_exp,
                         const double* prim_coeff,
                         const double* pt_bohr,
                         int backend,
                         double* v_out) {
  try {
    const auto basis = build_basis_from_c(nshell, sh_l, sh_center, sh_nprim, prim_exp, prim_coeff);
    const auto bk = (backend == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    const auto v = gpu4dft::snk_esp_matrix(basis, {pt_bohr[0], pt_bohr[1], pt_bohr[2]}, bk);
    std::copy(v.begin(), v.end(), v_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

int gpu4dft_xc_c_snk_compute_k(int nshell,
                               const int* sh_l,
                               const double* sh_center,
                               const int* sh_nprim,
                               const double* prim_exp,
                               const double* prim_coeff,
                               int npts,
                               const double* grid_coords,
                               const double* grid_weights,
                               int backend,
                               int precision,
                               int overlap_fitting,
                               int use_device_api,
                               const double* dm,
                               double* vk_out) {
  try {
    const auto basis = build_basis_from_c(nshell, sh_l, sh_center, sh_nprim, prim_exp, prim_coeff);
    const std::size_t un = static_cast<std::size_t>(npts);
    const auto grid = gpu4dft::MolecularGrid::from_points(
        std::span<const double>(grid_coords, un * 3),
        std::span<const double>(grid_weights, un),
        basis, 2048, 1e-13);
    gpu4dft::SnkOptions opts;
    opts.backend = (backend == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    opts.precision = (precision == 1) ? gpu4dft::Precision::FP32 : gpu4dft::Precision::FP64;
    opts.overlap_fitting = (overlap_fitting != 0);
    gpu4dft::SemiNumericalK snk(basis, grid, opts);

    if (use_device_api && opts.backend == gpu4dft::Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
      const std::size_t nao2 = static_cast<std::size_t>(basis.nao()) * basis.nao();
      double *d_dm = nullptr, *d_vk = nullptr;
      if (cudaMalloc(&d_dm, nao2 * sizeof(double)) != cudaSuccess) return -2;
      if (cudaMalloc(&d_vk, nao2 * sizeof(double)) != cudaSuccess) {
        cudaFree(d_dm);
        return -2;
      }
      cudaMemcpy(d_dm, dm, nao2 * sizeof(double), cudaMemcpyHostToDevice);
      snk.compute_k_device(d_dm, d_vk, nullptr);
      cudaMemcpy(vk_out, d_vk, nao2 * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(d_dm);
      cudaFree(d_vk);
      return 0;
#else
      return -1;
#endif
    }

    snk.compute_k(dm, vk_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

}  // extern "C"
