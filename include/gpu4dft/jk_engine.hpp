#pragma once

#include <memory>
#include <span>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/boys.hpp"
#include "gpu4dft/config.hpp"
#include "gpu4dft/harmonics.hpp"
#include "gpu4dft/linalg.hpp"
#include "gpu4dft/math.hpp"
#include "gpu4dft/one_electron.hpp"
#include "gpu4dft/rys.hpp"
#include "gpu4dft/shell_pairs.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

struct JKOptions {
  Backend backend = Backend::Cuda;
  Precision precision = Precision::FP64;
  GemmMode gemm_mode = GemmMode::FP64;
  int ozaki_slices = 8;
  double schwarz_cutoff = 1e-12;
  double fp32_threshold = 1e-6;
  bool compute_j = true;
  bool compute_k = true;
  double k_scale = 1.0;
  bool symmetrize_dm = true;
  double omega = 0.0;
  bool rsh_mode = false;
  double k_sr = 0.0;
  double k_lr = 0.0;
};

namespace detail {
struct DirectJKCudaCache;
struct RIJKCudaCache;
}  // namespace detail

// Direct on-the-fly Coulomb (J) and Exchange (K) engine without O(nao^4) ERI storage.
// Prebuilds and caches SoA shell pairs + Schwarz-sorted 2D tiles + Spherical<->Cartesian
// transforms on Host and GPU, supports Hermite J-engine contraction, Schwarz +
// density-weighted screening, incremental Fock updates, persistent GPU atomic work-stealing
// with thread-private J_ab tile accumulation, dynamic per-quartet FP64/FP32 mixed precision,
// and both symmetric SCF densities and non-symmetric TDDFT transition densities.
class DirectJKEngine {
 public:
  explicit DirectJKEngine(const Basis& basis, double prim_cutoff = 1e-14);
  ~DirectJKEngine();

  DirectJKEngine(DirectJKEngine&&) noexcept;
  DirectJKEngine& operator=(DirectJKEngine&&) noexcept;
  DirectJKEngine(const DirectJKEngine&) = delete;
  DirectJKEngine& operator=(const DirectJKEngine&) = delete;

  int nao() const { return basis_.nao(); }
  const Basis& basis() const { return basis_; }
  const std::vector<ShellPairData>& shell_pair_lists() const { return pair_lists_; }

  // Evaluates J and/or K (scaled by `opts.k_scale`) into row-major [nao x nao]
  // host buffers `vj_out` and `vk_out` (either pointer may be nullptr when disabled).
  void compute_jk(const double* dm, double* vj_out, double* vk_out,
                  const JKOptions& opts = {}) const;

  // Device-resident Direct J/K contraction on CUDA device pointers [nao x nao]
  // (`d_vj` and/or `d_vk` may be nullptr when disabled). Runs 100% on GPU with
  // persistent pre-allocated buffers and zero per-call cudaMalloc.
  void compute_jk_device(const double* d_dm, double* d_vj, double* d_vk,
                         const JKOptions& opts = {}, void* stream = nullptr) const;

  // Incremental Fock helper: evaluates Delta J(dm_new - dm_old) and
  // Delta K(dm_new - dm_old) using density-weighted Schwarz screening and
  // accumulates them into `vj_accum` and `vk_accum`.
  void compute_incremental_jk(const double* dm_new, const double* dm_old, double* vj_accum,
                              double* vk_accum, const JKOptions& opts = {}) const;

  // Open-shell UHF/UKS single-pass Coulomb J(D_alpha + D_beta) and Exchange K(D_alpha), K(D_beta)
  // (scaled by `opts.k_scale`). Any output pointer may be nullptr when disabled.
  void compute_jk_uhf(const double* dm_alpha, const double* dm_beta, double* vj_tot_out,
                      double* vk_alpha_out, double* vk_beta_out, const JKOptions& opts = {}) const;

  // Device-resident open-shell Direct J/K on CUDA device pointers [nao x nao].
  void compute_jk_uhf_device(const double* d_dm_alpha, const double* d_dm_beta, double* d_vj_tot,
                             double* d_vk_alpha, double* d_vk_beta, const JKOptions& opts = {},
                             void* stream = nullptr) const;

 private:
  Basis basis_;
  std::vector<ShellPairData> pair_lists_;
  std::vector<int> shell_cart_offset_;
  int ncart_total_ = 0;
  mutable detail::DirectJKCudaCache* cuda_cache_ = nullptr;
};

struct RIAuxChunk {
  int s0 = 0;
  int s1 = 0;
  int q0 = 0;
  int q1 = 0;
  int pc0 = 0;
  int pc1 = 0;
  std::vector<int> l_beg;
  std::vector<int> l_end;
};

struct RIHostMetric {
  bool built = false;
  bool is_cholesky = true;
  std::vector<double> factor;  // [naux * naux]: row-major L (is_cholesky) or V^{-1/2} (!is_cholesky)
};

// Resolution-of-the-Identity (Density Fitting) J/K engine using 2-center (P|Q) metric
// factorization and memory-bounded auxiliary-chunked + i-tiled occupied-space O(N_aux N_AO^2 N_occ)
// contractions for J, K, and analytical nuclear gradients (auxbasis_response=True), without
// allocating dense [N_aux, N_AO, N_AO] tensors in the default SCF/gradient path.
class RIJKEngine {
 public:
  RIJKEngine(const Basis& basis, const Basis& aux_basis, Backend backend = Backend::Cuda,
             double rsh_omega = 0.0);
  ~RIJKEngine();

  RIJKEngine(RIJKEngine&&) noexcept;
  RIJKEngine& operator=(RIJKEngine&&) noexcept;
  RIJKEngine(const RIJKEngine&) = delete;
  RIJKEngine& operator=(const RIJKEngine&) = delete;

  int nao() const { return basis_.nao(); }
  int naux() const { return aux_basis_.nao(); }
  double rsh_omega() const noexcept { return rsh_omega_; }

  bool has_device_b() const noexcept;
  const double* device_b_ptr() const noexcept;
  bool has_device_b_omega() const noexcept;
  const double* device_b_omega_ptr() const noexcept;

  // Lazy on-demand host [naux * nao * nao] B tensor builders for small unit tests.
  const std::vector<double>& b_tensor() const;
  const std::vector<double>& b_omega_tensor() const;

  void compute_jk(const double* dm, double* vj_out, double* vk_out,
                  const JKOptions& opts = {}) const;

  void compute_jk_device(const double* d_dm, double* d_vj, double* d_vk,
                         const JKOptions& opts = {}, void* stream = nullptr) const;

  // Memory-bounded occupied-space RI-J / RI-K contractions (with symmetric-DM fallback when
  // `c_occ` is nullptr, and `n_occ == 0` zero-K handling). `occ_scale` is 2.0 for RHF/RKS
  // and 1.0 for UHF/UKS or DM-factor modes.
  void compute_jk_mo(const double* dm, const double* c_occ, int n_occ, double occ_scale,
                     double* vj_out, double* vk_out, const JKOptions& opts = {}) const;

  void compute_jk_mo_device(const double* d_dm, const double* d_c_occ, int n_occ, double occ_scale,
                            double* d_vj, double* d_vk, const JKOptions& opts = {},
                            void* stream = nullptr) const;

  void compute_jk_uks_mo(const double* dm_alpha, const double* dm_beta,
                         const double* c_occ_alpha, int n_alpha, const double* c_occ_beta,
                         int n_beta, double* vj_tot_out, double* vk_alpha_out,
                         double* vk_beta_out, const JKOptions& opts = {}) const;

  void compute_jk_uks_mo_device(const double* d_dm_alpha, const double* d_dm_beta,
                                const double* d_c_occ_alpha, int n_alpha,
                                const double* d_c_occ_beta, int n_beta, double* d_vj_tot,
                                double* d_vk_alpha, double* d_vk_beta,
                                const JKOptions& opts = {}, void* stream = nullptr) const;

  // Memory-bounded analytical 2-electron RI-JK nuclear gradient [natom * 3] (Hartree / Bohr)
  // with auxbasis_response = true, covering RHF/RKS, UHF/UKS, pure GGA, hybrids, and RSH.
  std::vector<double> compute_ri_jk_grad(
      std::span<const Atom> atoms, const double* dm_alpha, const double* c_occ_alpha, int n_alpha,
      const double* dm_beta, const double* c_occ_beta, int n_beta, bool unrestricted,
      double j_fraction, double k_sr, double k_lr, double omega,
      Backend backend = Backend::Cuda) const;

  // Releases transient SCF streaming workspaces while keeping metric factorization & shell lists.
  void release_workspaces() const;

 private:
  void ensure_host_metric(double omega) const;
  void ensure_cuda_cache() const;

  Basis basis_;
  Basis aux_basis_;
  double rsh_omega_ = 0.0;
  std::vector<ShellPairData> orb_lists_;
  std::vector<ShellPairData> aux_lists_;  // indexed by lP in 0..aux_basis_.max_l()
  std::vector<int> orb_shell_cart_off_;
  std::vector<int> aux_shell_cart_off_;
  int ncart_orb_ = 0;
  int ncart_aux_ = 0;

  mutable RIHostMetric host_metric_0_;
  mutable RIHostMetric host_metric_w_;
  mutable std::vector<double> b_tensor_;        // lazy on-demand only
  mutable std::vector<double> b_omega_tensor_;  // lazy on-demand only
  mutable detail::RIJKCudaCache* cuda_cache_ = nullptr;
};

// Analytical 2-electron nuclear gradient [natom * 3] of:
//   E_2e = 0.5 * Tr(D * J(D)) - 0.25 * k_fraction * Tr(D * K(D)).
// Infers atom centers in order of first shell appearance in `basis`.
std::vector<double> jk_nuclear_grad(const Basis& basis, const double* dm, double k_fraction,
                                    Backend backend = Backend::Cuda,
                                    double schwarz_cutoff = 1e-12);

// Overload accepting an explicit atom list (handles basisless / ghost atoms).
std::vector<double> jk_nuclear_grad(const Basis& basis, std::span<const Atom> atoms,
                                    const double* dm, double k_fraction,
                                    Backend backend = Backend::Cuda,
                                    double schwarz_cutoff = 1e-12);

// Range-separated hybrid analytical 2-electron nuclear gradient [natom * 3] of:
//   E_2e = 0.5 * Tr(D * J_{1/r}(D))
//        - 0.25 * [k_sr * Tr(D * K_{1/r}(D)) + (k_lr - k_sr) * Tr(D * K_omega(D))].
std::vector<double> jk_nuclear_grad_rsh(const Basis& basis, const double* dm, double k_sr,
                                        double k_lr, double omega,
                                        Backend backend = Backend::Cuda,
                                        double schwarz_cutoff = 1e-12);

std::vector<double> jk_nuclear_grad_rsh(const Basis& basis, std::span<const Atom> atoms,
                                        const double* dm, double k_sr, double k_lr, double omega,
                                        Backend backend = Backend::Cuda,
                                        double schwarz_cutoff = 1e-12);

// Open-shell analytical 2-electron nuclear gradient [natom * 3] of:
//   E_2e = 0.5 * Tr(D_tot * J(D_tot)) - 0.5 * k_fraction * (Tr(D_a * K(D_a)) + Tr(D_b * K(D_b))).
std::vector<double> jk_nuclear_grad_uhf(const Basis& basis, const double* dm_alpha,
                                        const double* dm_beta, double k_fraction,
                                        Backend backend = Backend::Cuda,
                                        double schwarz_cutoff = 1e-12);

std::vector<double> jk_nuclear_grad_uhf(const Basis& basis, std::span<const Atom> atoms,
                                        const double* dm_alpha, const double* dm_beta,
                                        double k_fraction, Backend backend = Backend::Cuda,
                                        double schwarz_cutoff = 1e-12);

std::vector<double> jk_nuclear_grad_uhf_rsh(const Basis& basis, const double* dm_alpha,
                                            const double* dm_beta, double k_sr, double k_lr,
                                            double omega, Backend backend = Backend::Cuda,
                                            double schwarz_cutoff = 1e-12);

std::vector<double> jk_nuclear_grad_uhf_rsh(const Basis& basis, std::span<const Atom> atoms,
                                            const double* dm_alpha, const double* dm_beta,
                                            double k_sr, double k_lr, double omega,
                                            Backend backend = Backend::Cuda,
                                            double schwarz_cutoff = 1e-12);

namespace detail {

inline constexpr double kJKTwoOverSqrtPi = 1.12837916709551257390;

struct JKTaskTile {
  int bp;       // bra pair-list index
  int P;        // bra pair index in pair_lists[bp]
  int kp;       // ket pair-list index
  int Q_begin;  // start ket pair index
  int Q_end;    // end ket pair index (exclusive)
};

// Prebuilt GPU 2D tile (bra pair P, Schwarz-sorted ket slice Q_begin..Q_end)
// sorted in descending `max_schwarz = Q_ab * Q_cd[Q_begin]` order.
struct GpuJKTile {
  int bp;
  int P;
  int kp;
  int Q_begin;
  int Q_end;
  double max_schwarz;
};

struct GradQuartetTask {
  int bp;
  int P;
  int kp;
  int Q;
  double w_sym;
};

template <int Dim, int Deg, typename R>
GPU4DFT_HD inline void jk_hermite_1d_tiered(int max_a, int max_b, R inv_2p, R PA, R PB,
                                            R E[Dim][Dim][Deg], bool negate_odd) {
  const int max_t = max_a + max_b;
  for (int i = 0; i <= max_a; ++i)
    for (int j = 0; j <= max_b; ++j)
      for (int t = 0; t <= max_t; ++t) E[i][j][t] = R(0);
  E[0][0][0] = R(1);
  for (int i = 0; i < max_a; ++i) {
    for (int t = 0; t <= i + 1; ++t) {
      R val = PA * E[i][0][t];
      if (t > 0) val += inv_2p * E[i][0][t - 1];
      if (t + 1 <= i) val += R(t + 1) * E[i][0][t + 1];
      E[i + 1][0][t] = val;
    }
  }
  for (int i = 0; i <= max_a; ++i) {
    for (int j = 0; j < max_b; ++j) {
      const int deg = i + j;
      for (int t = 0; t <= deg + 1; ++t) {
        R val = PB * E[i][j][t];
        if (t > 0) val += inv_2p * E[i][j][t - 1];
        if (t + 1 <= deg) val += R(t + 1) * E[i][j][t + 1];
        E[i][j + 1][t] = val;
      }
    }
  }
  if (negate_odd) {
    for (int i = 0; i <= max_a; ++i)
      for (int j = 0; j <= max_b; ++j)
        for (int t = 1; t <= i + j; t += 2) E[i][j][t] = -E[i][j][t];
  }
}

template <typename R>
GPU4DFT_HD inline void jk_hermite_1d(int max_a, int max_b, R inv_2p, R PA, R PB,
                                     R E[5][5][9], bool negate_odd) {
  jk_hermite_1d_tiered<5, 9, R>(max_a, max_b, inv_2p, PA, PB, E, negate_odd);
}

// Tier-specialized Hermite J-engine path for pure Coulomb (J only) when L = la+lb+lc+ld <= 4.
template <int MaxL, typename R, bool SymmetrizeDM, typename AccumJBraFn, typename AccumJKetFn>
GPU4DFT_HD inline void eval_hermite_j_quartet_tiered(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_PQ, double w_J, const double* __restrict__ D_cart, int ncart_tot, int off_a,
    int off_b, int off_c, int off_d, AccumJBraFn accum_j_bra, AccumJKetFn accum_j_ket,
    double omega = 0.0) {
  constexpr int kMaxCart = (MaxL + 1) * (MaxL + 2) / 2;
  constexpr int kMaxCart2 = kMaxCart * kMaxCart;
  constexpr int kDim = MaxL + 1;
  constexpr int kPairDeg = (2 * MaxL < 4 ? 2 * MaxL : 4) + 1;

  if (la < 0 || la > MaxL || lb < 0 || lb > MaxL || lc < 0 || lc > MaxL || ld < 0 || ld > MaxL) {
    return;
  }
  const int lab = la + lb;
  const int lcd = lc + ld;
  const int L = lab + lcd;
  if (L > 4 || lab >= kPairDeg || lcd >= kPairDeg) return;
  const int nca = n_cart(la), ncb = n_cart(lb), ncc = n_cart(lc), ncd = n_cart(ld);

  CartTriple ta[kMaxCart], tb[kMaxCart], tc[kMaxCart], td[kMaxCart];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib_c = 0; ib_c < ncb; ++ib_c) tb[ib_c] = cart_triple(lb, ib_c);
  for (int ic = 0; ic < ncc; ++ic) tc[ic] = cart_triple(lc, ic);
  for (int id = 0; id < ncd; ++id) td[id] = cart_triple(ld, id);

  double j_ab[kMaxCart2] = {0.0};
  double j_cd[kMaxCart2] = {0.0};

  const R Ax = static_cast<R>(bra.Ax[ib]), Ay = static_cast<R>(bra.Ay[ib]),
          Az = static_cast<R>(bra.Az[ib]);
  const R Bx = static_cast<R>(bra.Bx[ib]), By = static_cast<R>(bra.By[ib]),
          Bz = static_cast<R>(bra.Bz[ib]);
  const R Cx = static_cast<R>(ket.Ax[ik]), Cy = static_cast<R>(ket.Ay[ik]),
          Cz = static_cast<R>(ket.Az[ik]);
  const R Dx = static_cast<R>(ket.Bx[ik]), Dy = static_cast<R>(ket.By[ik]),
          Dz = static_cast<R>(ket.Bz[ik]);

  const int bra_begin = bra.prim_offset[ib], bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik], ket_end = ket.prim_offset[ik + 1];

  for (int ip = bra_begin; ip < bra_end; ++ip) {
    const R p = static_cast<R>(bra.exponent[ip]);
    const R Px = static_cast<R>(bra.Px[ip]);
    const R Py = static_cast<R>(bra.Py[ip]);
    const R Pz = static_cast<R>(bra.Pz[ip]);
    const R Cab = static_cast<R>(bra.prefactor[ip]);
    const R inv_2p = R(0.5) / p;

    R Ex_ab[kDim][kDim][kPairDeg], Ey_ab[kDim][kDim][kPairDeg], Ez_ab[kDim][kDim][kPairDeg];
    jk_hermite_1d_tiered<kDim, kPairDeg, R>(la, lb, inv_2p, Px - Ax, Px - Bx, Ex_ab, false);
    jk_hermite_1d_tiered<kDim, kPairDeg, R>(la, lb, inv_2p, Py - Ay, Py - By, Ey_ab, false);
    jk_hermite_1d_tiered<kDim, kPairDeg, R>(la, lb, inv_2p, Pz - Az, Pz - Bz, Ez_ab, false);

    R D_bra_h[kPairDeg][kPairDeg][kPairDeg];
    R V_bra_h[kPairDeg][kPairDeg][kPairDeg];
    for (int t = 0; t <= lab; ++t)
      for (int u = 0; u <= lab - t; ++u)
        for (int v = 0; v <= lab - t - u; ++v) {
          D_bra_h[t][u][v] = R(0);
          V_bra_h[t][u][v] = R(0);
        }

    if (!same_PQ) {
      for (int ia = 0; ia < nca; ++ia) {
        const int ra = off_a + ia;
        for (int ib_c = 0; ib_c < ncb; ++ib_c) {
          const int rb = off_b + ib_c;
          const double d_ab_val =
              SymmetrizeDM
                  ? D_cart[ra * ncart_tot + rb]
                  : 0.5 * (D_cart[ra * ncart_tot + rb] + D_cart[rb * ncart_tot + ra]);
          const R dab = static_cast<R>(d_ab_val);
          for (int t = 0; t <= ta[ia].lx + tb[ib_c].lx; ++t) {
            const R ex = dab * Ex_ab[ta[ia].lx][tb[ib_c].lx][t];
            for (int u = 0; u <= ta[ia].ly + tb[ib_c].ly; ++u) {
              const R exy = ex * Ey_ab[ta[ia].ly][tb[ib_c].ly][u];
              for (int v = 0; v <= ta[ia].lz + tb[ib_c].lz; ++v) {
                D_bra_h[t][u][v] += exy * Ez_ab[ta[ia].lz][tb[ib_c].lz][v];
              }
            }
          }
        }
      }
    }

    for (int kp = ket_begin; kp < ket_end; ++kp) {
      const R q = static_cast<R>(ket.exponent[kp]);
      const R Qx = static_cast<R>(ket.Px[kp]);
      const R Qy = static_cast<R>(ket.Py[kp]);
      const R Qz = static_cast<R>(ket.Pz[kp]);
      const R Ccd = static_cast<R>(ket.prefactor[kp]);
      const R inv_2q = R(0.5) / q;

      R Ex_cd[kDim][kDim][kPairDeg], Ey_cd[kDim][kDim][kPairDeg], Ez_cd[kDim][kDim][kPairDeg];
      jk_hermite_1d_tiered<kDim, kPairDeg, R>(lc, ld, inv_2q, Qx - Cx, Qx - Dx, Ex_cd, true);
      jk_hermite_1d_tiered<kDim, kPairDeg, R>(lc, ld, inv_2q, Qy - Cy, Qy - Dy, Ey_cd, true);
      jk_hermite_1d_tiered<kDim, kPairDeg, R>(lc, ld, inv_2q, Qz - Cz, Qz - Dz, Ez_cd, true);

      R D_ket_h[kPairDeg][kPairDeg][kPairDeg];
      for (int tp = 0; tp <= lcd; ++tp)
        for (int up = 0; up <= lcd - tp; ++up)
          for (int vp = 0; vp <= lcd - tp - up; ++vp) D_ket_h[tp][up][vp] = R(0);

      for (int ic = 0; ic < ncc; ++ic) {
        const int rc = off_c + ic;
        for (int id = 0; id < ncd; ++id) {
          const int rd = off_d + id;
          const double d_cd_val =
              SymmetrizeDM
                  ? D_cart[rc * ncart_tot + rd]
                  : 0.5 * (D_cart[rc * ncart_tot + rd] + D_cart[rd * ncart_tot + rc]);
          const R dcd = static_cast<R>(d_cd_val);
          for (int tp = 0; tp <= tc[ic].lx + td[id].lx; ++tp) {
            const R ex = dcd * Ex_cd[tc[ic].lx][td[id].lx][tp];
            for (int up = 0; up <= tc[ic].ly + td[id].ly; ++up) {
              const R exy = ex * Ey_cd[tc[ic].ly][td[id].ly][up];
              for (int vp = 0; vp <= tc[ic].lz + td[id].lz; ++vp) {
                D_ket_h[tp][up][vp] += exy * Ez_cd[tc[ic].lz][td[id].lz][vp];
              }
            }
          }
        }
      }

      const R PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
      const R rho0 = p * q / (p + q);
      const R theta = omega_theta<R>(rho0, static_cast<R>(omega));
      const R rho = rho0 * theta;
      const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
      R F[5];
      boys_vec<R>(L, T, F);
      const R scale = Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho);

      R R_cur[5][5][5], R_nxt[5][5][5];
      for (int t = 0; t <= L; ++t)
        for (int u = 0; u <= L - t; ++u)
          for (int v = 0; v <= L - t - u; ++v) R_cur[t][u][v] = R(0);

      R pow_m2rho[5];
      pow_m2rho[0] = scale;
      for (int m = 1; m <= L; ++m) pow_m2rho[m] = pow_m2rho[m - 1] * (R(-2) * rho);

      R_cur[0][0][0] = pow_m2rho[L] * F[L];
      for (int m = L - 1; m >= 0; --m) {
        const int max_deg = L - m;
        R_nxt[0][0][0] = pow_m2rho[m] * F[m];
        for (int deg = 1; deg <= max_deg; ++deg) {
          for (int t = 0; t <= deg; ++t) {
            for (int u = 0; u <= deg - t; ++u) {
              const int v = deg - t - u;
              if (t > 0) {
                R_nxt[t][u][v] =
                    PQx * R_cur[t - 1][u][v] + (t > 1 ? R(t - 1) * R_cur[t - 2][u][v] : R(0));
              } else if (u > 0) {
                R_nxt[0][u][v] =
                    PQy * R_cur[0][u - 1][v] + (u > 1 ? R(u - 1) * R_cur[0][u - 2][v] : R(0));
              } else {
                R_nxt[0][0][v] =
                    PQz * R_cur[0][0][v - 1] + (v > 1 ? R(v - 1) * R_cur[0][0][v - 2] : R(0));
              }
            }
          }
        }
        for (int deg = 0; deg <= max_deg; ++deg)
          for (int t = 0; t <= deg; ++t)
            for (int u = 0; u <= deg - t; ++u) {
              const int v = deg - t - u;
              R_cur[t][u][v] = R_nxt[t][u][v];
            }
      }

      R V_ket_h[kPairDeg][kPairDeg][kPairDeg];
      if (!same_PQ) {
        for (int tp = 0; tp <= lcd; ++tp)
          for (int up = 0; up <= lcd - tp; ++up)
            for (int vp = 0; vp <= lcd - tp - up; ++vp) V_ket_h[tp][up][vp] = R(0);
      }

      for (int t = 0; t <= lab; ++t) {
        for (int u = 0; u <= lab - t; ++u) {
          for (int v = 0; v <= lab - t - u; ++v) {
            R sum_k = R(0);
            const R db = D_bra_h[t][u][v];
            for (int tp = 0; tp <= lcd; ++tp) {
              for (int up = 0; up <= lcd - tp; ++up) {
                for (int vp = 0; vp <= lcd - tp - up; ++vp) {
                  const R r_val = R_cur[t + tp][u + up][v + vp];
                  sum_k += r_val * D_ket_h[tp][up][vp];
                  if (!same_PQ) {
                    V_ket_h[tp][up][vp] += r_val * db;
                  }
                }
              }
            }
            V_bra_h[t][u][v] += sum_k;
          }
        }
      }

      if (!same_PQ) {
        for (int ic = 0; ic < ncc; ++ic) {
          for (int id = 0; id < ncd; ++id) {
            R sum = R(0);
            for (int tp = 0; tp <= tc[ic].lx + td[id].lx; ++tp) {
              const R ex = Ex_cd[tc[ic].lx][td[id].lx][tp];
              for (int up = 0; up <= tc[ic].ly + td[id].ly; ++up) {
                const R exy = ex * Ey_cd[tc[ic].ly][td[id].ly][up];
                for (int vp = 0; vp <= tc[ic].lz + td[id].lz; ++vp) {
                  sum += exy * Ez_cd[tc[ic].lz][td[id].lz][vp] * V_ket_h[tp][up][vp];
                }
              }
            }
            j_cd[ic * ncd + id] += static_cast<double>(sum);
          }
        }
      }
    }

    for (int ia = 0; ia < nca; ++ia) {
      for (int ib_c = 0; ib_c < ncb; ++ib_c) {
        R sum = R(0);
        for (int t = 0; t <= ta[ia].lx + tb[ib_c].lx; ++t) {
          const R ex = Ex_ab[ta[ia].lx][tb[ib_c].lx][t];
          for (int u = 0; u <= ta[ia].ly + tb[ib_c].ly; ++u) {
            const R exy = ex * Ey_ab[ta[ia].ly][tb[ib_c].ly][u];
            for (int v = 0; v <= ta[ia].lz + tb[ib_c].lz; ++v) {
              sum += exy * Ez_ab[ta[ia].lz][tb[ib_c].lz][v] * V_bra_h[t][u][v];
            }
          }
        }
        j_ab[ia * ncb + ib_c] += static_cast<double>(sum);
      }
    }
  }

  for (int ia = 0; ia < nca; ++ia) {
    for (int ib_c = 0; ib_c < ncb; ++ib_c) {
      accum_j_bra(ia, ib_c, w_J * j_ab[ia * ncb + ib_c]);
    }
  }
  if (!same_PQ) {
    for (int ic = 0; ic < ncc; ++ic) {
      for (int id = 0; id < ncd; ++id) {
        accum_j_ket(off_c + ic, off_d + id, w_J * j_cd[ic * ncd + id]);
      }
    }
  }
}

template <typename R, typename AccumFn>
GPU4DFT_HD inline void eval_hermite_j_quartet(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_PQ, double w_J, const double* D_cart, int ncart_tot, int off_a, int off_b,
    int off_c, int off_d, AccumFn accum_j, double omega = 0.0) {
  auto accum_bra = [&](int ia, int ib_c, double val) { accum_j(off_a + ia, off_b + ib_c, val); };
  eval_hermite_j_quartet_tiered<4, R, true>(la, lb, lc, ld, bra, ib, ket, ik, same_PQ, w_J, D_cart,
                                            ncart_tot, off_a, off_b, off_c, off_d, accum_bra,
                                            accum_j, omega);
}

template <int MaxL, typename R>
GPU4DFT_HD inline void jk_rys_1d_tiered(int la, int lb, int lc, int ld, R C00, R D00, R B10, R B01,
                                        R B00, R AB, R CD,
                                        R I[MaxL + 1][MaxL + 1][MaxL + 1][MaxL + 1]) {
  constexpr int kDim = MaxL + 1;
  constexpr int kDeg = 2 * MaxL + 1;
  if (la < 0 || la > MaxL || lb < 0 || lb > MaxL || lc < 0 || lc > MaxL || ld < 0 || ld > MaxL) {
    return;
  }
  const int lab = la + lb;
  const int lcd = lc + ld;
  R G[kDeg][kDeg];
  G[0][0] = R(1);
  for (int n = 0; n < lab; ++n) {
    G[n + 1][0] = C00 * G[n][0] + (n > 0 ? R(n) * B10 * G[n - 1][0] : R(0));
  }
  for (int m = 0; m < lcd; ++m) {
    for (int n = 0; n <= lab; ++n) {
      R val = D00 * G[n][m];
      if (m > 0) val += R(m) * B01 * G[n][m - 1];
      if (n > 0) val += R(n) * B00 * G[n - 1][m];
      G[n][m + 1] = val;
    }
  }

  R H[kDim][kDim][kDeg];
  R cur[kDeg][kDeg];
  for (int n = 0; n <= lab; ++n)
    for (int m = 0; m <= lcd; ++m) cur[n][m] = G[n][m];
  for (int j = 0; j <= lb; ++j) {
    for (int i = 0; i <= la; ++i)
      for (int m = 0; m <= lcd; ++m) H[i][j][m] = cur[i][m];
    if (j < lb) {
      for (int i = 0; i <= lab - j - 1; ++i)
        for (int m = 0; m <= lcd; ++m) cur[i][m] = cur[i + 1][m] + AB * cur[i][m];
    }
  }

  for (int i = 0; i <= la; ++i) {
    for (int j = 0; j <= lb; ++j) {
      R kcur[kDeg];
      for (int m = 0; m <= lcd; ++m) kcur[m] = H[i][j][m];
      for (int l = 0; l <= ld; ++l) {
        for (int k = 0; k <= lc; ++k) I[i][j][k][l] = kcur[k];
        if (l < ld) {
          for (int k = 0; k <= lcd - l - 1; ++k) kcur[k] = kcur[k + 1] + CD * kcur[k];
        }
      }
    }
  }
}

template <typename R>
GPU4DFT_HD inline void jk_rys_1d(int la, int lb, int lc, int ld, R C00, R D00, R B10, R B01,
                                 R B00, R AB, R CD, R I[5][5][5][5]) {
  jk_rys_1d_tiered<4, R>(la, lb, lc, ld, C00, D00, B10, B01, B00, AB, CD, I);
}

// Pure Rys Direct J + K quartet contraction (NO Hermite J branch inside, keeping
// CUDA kernel register and stack-frame footprints minimal and spill-free!).
template <int MaxL, typename R, bool SymmetrizeDM, typename AccumJBraFn, typename AccumJKetFn,
          typename AccumKFn>
GPU4DFT_HD inline void eval_rys_jk_quartet_tiered(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_PQ, bool do_j, bool do_k, double w_J, double w_K,
    const double* __restrict__ D_cart, int ncart_tot, int off_a, int off_b, int off_c, int off_d,
    AccumJBraFn accum_j_bra, AccumJKetFn accum_j_ket, AccumKFn accum_k, double omega = 0.0) {
  if (la < 0 || la > MaxL || lb < 0 || lb > MaxL || lc < 0 || lc > MaxL || ld < 0 || ld > MaxL) {
    return;
  }

  constexpr int kMaxCart = (MaxL + 1) * (MaxL + 2) / 2;
  constexpr int kMaxCart2 = kMaxCart * kMaxCart;
  constexpr int kDim = MaxL + 1;
  constexpr int kMaxRoots = 2 * MaxL + 1;

  const int bra_begin = bra.prim_offset[ib], bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik], ket_end = ket.prim_offset[ik + 1];

  // 1. Dedicated (ss|ss) scalar fast path.
  if (la == 0 && lb == 0 && lc == 0 && ld == 0) {
    double val = 0.0;
    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const R p = static_cast<R>(bra.exponent[ip]);
      const R Px = static_cast<R>(bra.Px[ip]);
      const R Py = static_cast<R>(bra.Py[ip]);
      const R Pz = static_cast<R>(bra.Pz[ip]);
      const R Cab = static_cast<R>(bra.prefactor[ip]);
      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const R q = static_cast<R>(ket.exponent[kp]);
        const R PQx = Px - static_cast<R>(ket.Px[kp]);
        const R PQy = Py - static_cast<R>(ket.Py[kp]);
        const R PQz = Pz - static_cast<R>(ket.Pz[kp]);
        const R Ccd = static_cast<R>(ket.prefactor[kp]);
        const R rho = p * q / (p + q);
        const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
        val += static_cast<double>(Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho) *
                                   boys_f0_erf(T, rho, static_cast<R>(omega)));
      }
    }
    if (do_j) {
      const double d_cd =
          SymmetrizeDM
              ? D_cart[off_c * ncart_tot + off_d]
              : 0.5 * (D_cart[off_c * ncart_tot + off_d] + D_cart[off_d * ncart_tot + off_c]);
      accum_j_bra(0, 0, w_J * val * d_cd);
      if (!same_PQ) {
        const double d_ab =
            SymmetrizeDM
                ? D_cart[off_a * ncart_tot + off_b]
                : 0.5 * (D_cart[off_a * ncart_tot + off_b] + D_cart[off_b * ncart_tot + off_a]);
        accum_j_ket(off_c, off_d, w_J * val * d_ab);
      }
    }
    if (do_k) {
      const double wk = w_K * val;
      accum_k(off_a, off_c, wk * D_cart[off_b * ncart_tot + off_d]);
      accum_k(off_b, off_c, wk * D_cart[off_a * ncart_tot + off_d]);
      accum_k(off_a, off_d, wk * D_cart[off_b * ncart_tot + off_c]);
      accum_k(off_b, off_d, wk * D_cart[off_a * ncart_tot + off_c]);
      if constexpr (!SymmetrizeDM) {
        if (!same_PQ) {
          accum_k(off_c, off_a, wk * D_cart[off_d * ncart_tot + off_b]);
          accum_k(off_c, off_b, wk * D_cart[off_d * ncart_tot + off_a]);
          accum_k(off_d, off_a, wk * D_cart[off_c * ncart_tot + off_b]);
          accum_k(off_d, off_b, wk * D_cart[off_c * ncart_tot + off_a]);
        }
      }
    }
    return;
  }

  const int nca = n_cart(la), ncb = n_cart(lb), ncc = n_cart(lc), ncd = n_cart(ld);
  const int n_roots = (la + lb + lc + ld) / 2 + 1;
  const double Ax = bra.Ax[ib], Ay = bra.Ay[ib], Az = bra.Az[ib];
  const double Bx = bra.Bx[ib], By = bra.By[ib], Bz = bra.Bz[ib];
  const double Cx = ket.Ax[ik], Cy = ket.Ay[ik], Cz = ket.Az[ik];
  const double Dx = ket.Bx[ik], Dy = ket.By[ik], Dz = ket.Bz[ik];
  const R ABx = static_cast<R>(Ax - Bx), ABy = static_cast<R>(Ay - By),
          ABz = static_cast<R>(Az - Bz);
  const R CDx = static_cast<R>(Cx - Dx), CDy = static_cast<R>(Cy - Dy),
          CDz = static_cast<R>(Cz - Dz);

  CartTriple ta[kMaxCart], tb[kMaxCart], tc[kMaxCart], td[kMaxCart];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib_c = 0; ib_c < ncb; ++ib_c) tb[ib_c] = cart_triple(lb, ib_c);
  for (int ic = 0; ic < ncc; ++ic) tc[ic] = cart_triple(lc, ic);
  for (int id = 0; id < ncd; ++id) td[id] = cart_triple(ld, id);

  // 2. Contracted-ERI-first fast path ONLY for `sp` (MaxL == 1, <= 81 scalars,
  //    covering >70% of all quartets with zero fallback code bloat!).
  if constexpr (MaxL == 1) {
    R eri_cart[81];
    const int n_comp = nca * ncb * ncc * ncd;
    for (int idx = 0; idx < n_comp; ++idx) eri_cart[idx] = R(0);

    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const R p = static_cast<R>(bra.exponent[ip]);
      const R Px = static_cast<R>(bra.Px[ip]);
      const R Py = static_cast<R>(bra.Py[ip]);
      const R Pz = static_cast<R>(bra.Pz[ip]);
      const R Cab = static_cast<R>(bra.prefactor[ip]);
      const R PAx = Px - static_cast<R>(Ax);
      const R PAy = Py - static_cast<R>(Ay);
      const R PAz = Pz - static_cast<R>(Az);
      const R inv_2p = R(0.5) / p;

      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const R q = static_cast<R>(ket.exponent[kp]);
        const R Qx = static_cast<R>(ket.Px[kp]);
        const R Qy = static_cast<R>(ket.Py[kp]);
        const R Qz = static_cast<R>(ket.Pz[kp]);
        const R Ccd = static_cast<R>(ket.prefactor[kp]);
        const R PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
        const R pq_sum = p + q;
        const R rho = p * q / pq_sum;
        const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
        const R scale = Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho);

        R u_roots[kMaxRoots];
        R w_weights[kMaxRoots];
        rys_roots_weights_erf<R>(n_roots, T, rho, static_cast<R>(omega), u_roots, w_weights);

        const R inv_2q = R(0.5) / q;
        const R inv_2pq = R(0.5) / pq_sum;
        const R rho_over_p = rho / p;
        const R rho_over_q = rho / q;
        const R QCx = Qx - static_cast<R>(Cx);
        const R QCy = Qy - static_cast<R>(Cy);
        const R QCz = Qz - static_cast<R>(Cz);

        for (int r = 0; r < n_roots; ++r) {
          const R u = u_roots[r];
          const R Wr = scale * w_weights[r];
          const R B00 = inv_2pq * u;
          const R B10 = inv_2p * (R(1) - rho_over_p * u);
          const R B01 = inv_2q * (R(1) - rho_over_q * u);

          R Ix[kDim][kDim][kDim][kDim], Iy[kDim][kDim][kDim][kDim], Iz[kDim][kDim][kDim][kDim];
          jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAx - rho_over_p * u * PQx,
                                    QCx + rho_over_q * u * PQx, B10, B01, B00, ABx, CDx, Ix);
          jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAy - rho_over_p * u * PQy,
                                    QCy + rho_over_q * u * PQy, B10, B01, B00, ABy, CDy, Iy);
          jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAz - rho_over_p * u * PQz,
                                    QCz + rho_over_q * u * PQz, B10, B01, B00, ABz, CDz, Iz);

          int out_idx = 0;
          for (int ia = 0; ia < nca; ++ia) {
            for (int ib_c = 0; ib_c < ncb; ++ib_c) {
              for (int ic = 0; ic < ncc; ++ic) {
                for (int id = 0; id < ncd; ++id) {
                  eri_cart[out_idx++] +=
                      Wr * Ix[ta[ia].lx][tb[ib_c].lx][tc[ic].lx][td[id].lx] *
                      Iy[ta[ia].ly][tb[ib_c].ly][tc[ic].ly][td[id].ly] *
                      Iz[ta[ia].lz][tb[ib_c].lz][tc[ic].lz][td[id].lz];
                }
              }
            }
          }
        }
      }
    }

    const int ncd_blk = ncc * ncd;
    if (do_j) {
      R d_cd_sym[kMaxCart2];
      for (int ic = 0; ic < ncc; ++ic) {
        const int rc = off_c + ic;
        for (int id = 0; id < ncd; ++id) {
          const int rd = off_d + id;
          d_cd_sym[ic * ncd + id] = static_cast<R>(
              SymmetrizeDM
                  ? D_cart[rc * ncart_tot + rd]
                  : 0.5 * (D_cart[rc * ncart_tot + rd] + D_cart[rd * ncart_tot + rc]));
        }
      }
      R j_cd_loc[kMaxCart2] = {R(0)};
      for (int ia = 0; ia < nca; ++ia) {
        const int ra = off_a + ia;
        for (int ib_c = 0; ib_c < ncb; ++ib_c) {
          const int rb = off_b + ib_c;
          const R dab_sym = static_cast<R>(
              SymmetrizeDM
                  ? D_cart[ra * ncart_tot + rb]
                  : 0.5 * (D_cart[ra * ncart_tot + rb] + D_cart[rb * ncart_tot + ra]));
          const R* e_ab = eri_cart + (ia * ncb + ib_c) * ncd_blk;
          R sum_jab = R(0);
          for (int icd = 0; icd < ncd_blk; ++icd) {
            const R v = e_ab[icd];
            sum_jab += v * d_cd_sym[icd];
            if (!same_PQ) j_cd_loc[icd] += v * dab_sym;
          }
          accum_j_bra(ia, ib_c, w_J * static_cast<double>(sum_jab));
        }
      }
      if (!same_PQ) {
        for (int ic = 0; ic < ncc; ++ic)
          for (int id = 0; id < ncd; ++id)
            accum_j_ket(off_c + ic, off_d + id, w_J * static_cast<double>(j_cd_loc[ic * ncd + id]));
      }
    }

    if (do_k) {
      R k_ac[kMaxCart2] = {R(0)};
      R k_bc[kMaxCart2] = {R(0)};
      R k_ad[kMaxCart2] = {R(0)};
      R k_bd[kMaxCart2] = {R(0)};
      R k_ca[SymmetrizeDM ? 1 : kMaxCart2] = {R(0)};
      R k_cb[SymmetrizeDM ? 1 : kMaxCart2] = {R(0)};
      R k_da[SymmetrizeDM ? 1 : kMaxCart2] = {R(0)};
      R k_db[SymmetrizeDM ? 1 : kMaxCart2] = {R(0)};

      for (int ia = 0; ia < nca; ++ia) {
        const int ra = off_a + ia;
        for (int ib_c = 0; ib_c < ncb; ++ib_c) {
          const int rb = off_b + ib_c;
          const R* e_ab = eri_cart + (ia * ncb + ib_c) * ncd_blk;
          for (int ic = 0; ic < ncc; ++ic) {
            const int rc = off_c + ic;
            const R d_ac = static_cast<R>(D_cart[ra * ncart_tot + rc]);
            const R d_bc = static_cast<R>(D_cart[rb * ncart_tot + rc]);
            const R d_ca = SymmetrizeDM ? R(0) : static_cast<R>(D_cart[rc * ncart_tot + ra]);
            const R d_cb = SymmetrizeDM ? R(0) : static_cast<R>(D_cart[rc * ncart_tot + rb]);
            for (int id = 0; id < ncd; ++id) {
              const int rd = off_d + id;
              const R v = e_ab[ic * ncd + id];
              k_ac[ia * ncc + ic] += v * static_cast<R>(D_cart[rb * ncart_tot + rd]);
              k_bc[ib_c * ncc + ic] += v * static_cast<R>(D_cart[ra * ncart_tot + rd]);
              k_ad[ia * ncd + id] += v * d_bc;
              k_bd[ib_c * ncd + id] += v * d_ac;
              if constexpr (!SymmetrizeDM) {
                if (!same_PQ) {
                  k_ca[ic * nca + ia] += v * static_cast<R>(D_cart[rd * ncart_tot + rb]);
                  k_cb[ic * ncb + ib_c] += v * static_cast<R>(D_cart[rd * ncart_tot + ra]);
                  k_da[id * nca + ia] += v * d_cb;
                  k_db[id * ncb + ib_c] += v * d_ca;
                }
              }
            }
          }
        }
      }

      for (int ia = 0; ia < nca; ++ia)
        for (int ic = 0; ic < ncc; ++ic)
          accum_k(off_a + ia, off_c + ic, w_K * static_cast<double>(k_ac[ia * ncc + ic]));
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        for (int ic = 0; ic < ncc; ++ic)
          accum_k(off_b + ib_c, off_c + ic, w_K * static_cast<double>(k_bc[ib_c * ncc + ic]));
      for (int ia = 0; ia < nca; ++ia)
        for (int id = 0; id < ncd; ++id)
          accum_k(off_a + ia, off_d + id, w_K * static_cast<double>(k_ad[ia * ncd + id]));
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        for (int id = 0; id < ncd; ++id)
          accum_k(off_b + ib_c, off_d + id, w_K * static_cast<double>(k_bd[ib_c * ncd + id]));

      if constexpr (!SymmetrizeDM) {
        if (!same_PQ) {
          for (int ic = 0; ic < ncc; ++ic)
            for (int ia = 0; ia < nca; ++ia)
              accum_k(off_c + ic, off_a + ia, w_K * static_cast<double>(k_ca[ic * nca + ia]));
          for (int ic = 0; ic < ncc; ++ic)
            for (int ib_c = 0; ib_c < ncb; ++ib_c)
              accum_k(off_c + ic, off_b + ib_c, w_K * static_cast<double>(k_cb[ic * ncb + ib_c]));
          for (int id = 0; id < ncd; ++id)
            for (int ia = 0; ia < nca; ++ia)
              accum_k(off_d + id, off_a + ia, w_K * static_cast<double>(k_da[id * nca + ia]));
          for (int id = 0; id < ncd; ++id)
            for (int ib_c = 0; ib_c < ncb; ++ib_c)
              accum_k(off_d + id, off_b + ib_c, w_K * static_cast<double>(k_db[id * ncb + ib_c]));
        }
      }
    }
    return;
  }

  // 3. Single lean direct contraction path for MaxL >= 2 (d, f, g): uses `R` (`float` in FP32/Mixed
  //    tail, `double` in FP64), casting to `double` on global accumulation!
  R j_ab[kMaxCart2] = {R(0)};
  R j_cd[kMaxCart2] = {R(0)};
  R k_ac[kMaxCart2] = {R(0)};
  R k_bc[kMaxCart2] = {R(0)};
  R k_ad[kMaxCart2] = {R(0)};
  R k_bd[kMaxCart2] = {R(0)};
  R k_ca[SymmetrizeDM ? 1 : kMaxCart2] = {R(0)};
  R k_cb[SymmetrizeDM ? 1 : kMaxCart2] = {R(0)};
  R k_da[SymmetrizeDM ? 1 : kMaxCart2] = {R(0)};
  R k_db[SymmetrizeDM ? 1 : kMaxCart2] = {R(0)};

  for (int ip = bra_begin; ip < bra_end; ++ip) {
    const R p = static_cast<R>(bra.exponent[ip]);
    const R Px = static_cast<R>(bra.Px[ip]);
    const R Py = static_cast<R>(bra.Py[ip]);
    const R Pz = static_cast<R>(bra.Pz[ip]);
    const R Cab = static_cast<R>(bra.prefactor[ip]);
    const R PAx = Px - static_cast<R>(Ax);
    const R PAy = Py - static_cast<R>(Ay);
    const R PAz = Pz - static_cast<R>(Az);
    const R inv_2p = R(0.5) / p;

    for (int kp = ket_begin; kp < ket_end; ++kp) {
      const R q = static_cast<R>(ket.exponent[kp]);
      const R Qx = static_cast<R>(ket.Px[kp]);
      const R Qy = static_cast<R>(ket.Py[kp]);
      const R Qz = static_cast<R>(ket.Pz[kp]);
      const R Ccd = static_cast<R>(ket.prefactor[kp]);
      const R PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
      const R pq_sum = p + q;
      const R rho = p * q / pq_sum;
      const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
      const R scale = Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho);

      R u_roots[kMaxRoots];
      R w_weights[kMaxRoots];
      rys_roots_weights_erf<R>(n_roots, T, rho, static_cast<R>(omega), u_roots, w_weights);

      const R inv_2q = R(0.5) / q;
      const R inv_2pq = R(0.5) / pq_sum;
      const R rho_over_p = rho / p;
      const R rho_over_q = rho / q;
      const R QCx = Qx - static_cast<R>(Cx);
      const R QCy = Qy - static_cast<R>(Cy);
      const R QCz = Qz - static_cast<R>(Cz);

      for (int r = 0; r < n_roots; ++r) {
        const R u = u_roots[r];
        const R Wr = scale * w_weights[r];
        const R B00 = inv_2pq * u;
        const R B10 = inv_2p * (R(1) - rho_over_p * u);
        const R B01 = inv_2q * (R(1) - rho_over_q * u);

        R Ix[kDim][kDim][kDim][kDim], Iy[kDim][kDim][kDim][kDim], Iz[kDim][kDim][kDim][kDim];
        jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAx - rho_over_p * u * PQx,
                                  QCx + rho_over_q * u * PQx, B10, B01, B00, ABx, CDx, Ix);
        jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAy - rho_over_p * u * PQy,
                                  QCy + rho_over_q * u * PQy, B10, B01, B00, ABy, CDy, Iy);
        jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAz - rho_over_p * u * PQz,
                                  QCz + rho_over_q * u * PQz, B10, B01, B00, ABz, CDz, Iz);

        for (int ia = 0; ia < nca; ++ia) {
          const int ra = off_a + ia;
          for (int ib_c = 0; ib_c < ncb; ++ib_c) {
            const int rb = off_b + ib_c;
            const R dab = static_cast<R>(
                SymmetrizeDM
                    ? D_cart[ra * ncart_tot + rb]
                    : 0.5 * (D_cart[ra * ncart_tot + rb] + D_cart[rb * ncart_tot + ra]));
            R sum_jab = R(0);
            for (int ic = 0; ic < ncc; ++ic) {
              const int rc = off_c + ic;
              const R dac = static_cast<R>(D_cart[ra * ncart_tot + rc]);
              const R dbc = static_cast<R>(D_cart[rb * ncart_tot + rc]);
              const R dca = SymmetrizeDM ? R(0) : static_cast<R>(D_cart[rc * ncart_tot + ra]);
              const R dcb = SymmetrizeDM ? R(0) : static_cast<R>(D_cart[rc * ncart_tot + rb]);
              for (int id = 0; id < ncd; ++id) {
                const int rd = off_d + id;
                const R v = Wr * Ix[ta[ia].lx][tb[ib_c].lx][tc[ic].lx][td[id].lx] *
                            Iy[ta[ia].ly][tb[ib_c].ly][tc[ic].ly][td[id].ly] *
                            Iz[ta[ia].lz][tb[ib_c].lz][tc[ic].lz][td[id].lz];
                if (do_j) {
                  const R dcd_s = static_cast<R>(
                      SymmetrizeDM
                          ? D_cart[rc * ncart_tot + rd]
                          : 0.5 * (D_cart[rc * ncart_tot + rd] + D_cart[rd * ncart_tot + rc]));
                  sum_jab += v * dcd_s;
                  if (!same_PQ) j_cd[ic * ncd + id] += v * dab;
                }
                if (do_k) {
                  k_ac[ia * ncc + ic] += v * static_cast<R>(D_cart[rb * ncart_tot + rd]);
                  k_bc[ib_c * ncc + ic] += v * static_cast<R>(D_cart[ra * ncart_tot + rd]);
                  k_ad[ia * ncd + id] += v * dbc;
                  k_bd[ib_c * ncd + id] += v * dac;
                  if constexpr (!SymmetrizeDM) {
                    if (!same_PQ) {
                      k_ca[ic * nca + ia] += v * static_cast<R>(D_cart[rd * ncart_tot + rb]);
                      k_cb[ic * ncb + ib_c] += v * static_cast<R>(D_cart[rd * ncart_tot + ra]);
                      k_da[id * nca + ia] += v * dcb;
                      k_db[id * ncb + ib_c] += v * dca;
                    }
                  }
                }
              }
            }
            if (do_j) j_ab[ia * ncb + ib_c] += sum_jab;
          }
        }
      }
    }
  }

  if (do_j) {
    for (int ia = 0; ia < nca; ++ia)
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        accum_j_bra(ia, ib_c, w_J * static_cast<double>(j_ab[ia * ncb + ib_c]));
    if (!same_PQ) {
      for (int ic = 0; ic < ncc; ++ic)
        for (int id = 0; id < ncd; ++id)
          accum_j_ket(off_c + ic, off_d + id, w_J * static_cast<double>(j_cd[ic * ncd + id]));
    }
  }
  if (do_k) {
    for (int ia = 0; ia < nca; ++ia)
      for (int ic = 0; ic < ncc; ++ic)
        accum_k(off_a + ia, off_c + ic, w_K * static_cast<double>(k_ac[ia * ncc + ic]));
    for (int ib_c = 0; ib_c < ncb; ++ib_c)
      for (int ic = 0; ic < ncc; ++ic)
        accum_k(off_b + ib_c, off_c + ic, w_K * static_cast<double>(k_bc[ib_c * ncc + ic]));
    for (int ia = 0; ia < nca; ++ia)
      for (int id = 0; id < ncd; ++id)
        accum_k(off_a + ia, off_d + id, w_K * static_cast<double>(k_ad[ia * ncd + id]));
    for (int ib_c = 0; ib_c < ncb; ++ib_c)
      for (int id = 0; id < ncd; ++id)
        accum_k(off_b + ib_c, off_d + id, w_K * static_cast<double>(k_bd[ib_c * ncd + id]));

    if constexpr (!SymmetrizeDM) {
      if (!same_PQ) {
        for (int ic = 0; ic < ncc; ++ic)
          for (int ia = 0; ia < nca; ++ia)
            accum_k(off_c + ic, off_a + ia, w_K * static_cast<double>(k_ca[ic * nca + ia]));
        for (int ic = 0; ic < ncc; ++ic)
          for (int ib_c = 0; ib_c < ncb; ++ib_c)
            accum_k(off_c + ic, off_b + ib_c, w_K * static_cast<double>(k_cb[ic * ncb + ib_c]));
        for (int id = 0; id < ncd; ++id)
          for (int ia = 0; ia < nca; ++ia)
            accum_k(off_d + id, off_a + ia, w_K * static_cast<double>(k_da[id * nca + ia]));
        for (int id = 0; id < ncd; ++id)
          for (int ib_c = 0; ib_c < ncb; ++ib_c)
            accum_k(off_d + id, off_b + ib_c, w_K * static_cast<double>(k_db[id * ncb + ib_c]));
      }
    }
  }
}

// Host/general Direct J + K quartet contraction (dispatches pure-J L<=4 to Hermite, else Rys).
template <int MaxL, typename R, bool SymmetrizeDM, typename AccumJBraFn, typename AccumJKetFn,
          typename AccumKFn>
GPU4DFT_HD inline void eval_direct_jk_quartet_tiered(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_PQ, bool do_j, bool do_k, double w_J, double w_K,
    const double* __restrict__ D_cart, int ncart_tot, int off_a, int off_b, int off_c, int off_d,
    AccumJBraFn accum_j_bra, AccumJKetFn accum_j_ket, AccumKFn accum_k, double omega = 0.0) {
  if (la < 0 || la > MaxL || lb < 0 || lb > MaxL || lc < 0 || lc > MaxL || ld < 0 || ld > MaxL) {
    return;
  }
  if (do_j && !do_k && (la + lb + lc + ld) <= 4) {
    eval_hermite_j_quartet_tiered<MaxL, R, SymmetrizeDM>(
        la, lb, lc, ld, bra, ib, ket, ik, same_PQ, w_J, D_cart, ncart_tot, off_a, off_b, off_c,
        off_d, accum_j_bra, accum_j_ket, omega);
    return;
  }
  eval_rys_jk_quartet_tiered<MaxL, R, SymmetrizeDM>(
      la, lb, lc, ld, bra, ib, ket, ik, same_PQ, do_j, do_k, w_J, w_K, D_cart, ncart_tot, off_a,
      off_b, off_c, off_d, accum_j_bra, accum_j_ket, accum_k, omega);
}

template <typename R, typename AccumJFn, typename AccumKFn>
GPU4DFT_HD inline void eval_direct_jk_quartet(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_PQ, bool do_j, bool do_k, double w_J, double w_K, const double* D_cart,
    int ncart_tot, int off_a, int off_b, int off_c, int off_d, AccumJFn accum_j,
    AccumKFn accum_k, double omega = 0.0) {
  auto accum_bra = [&](int ia, int ib_c, double val) { accum_j(off_a + ia, off_b + ib_c, val); };
  eval_direct_jk_quartet_tiered<4, R, true>(la, lb, lc, ld, bra, ib, ket, ik, same_PQ, do_j, do_k,
                                            w_J, w_K, D_cart, ncart_tot, off_a, off_b, off_c, off_d,
                                            accum_bra, accum_j, accum_k, omega);
}

// Single-pass UHF/UKS Rys Direct J(D_tot) + K(D_alpha) + K(D_beta) contraction
// for symmetric spin density matrices (`SymmetrizeDM = true`).
template <int MaxL, typename R, typename AccumJBraFn, typename AccumJKetFn, typename AccumKaFn,
          typename AccumKbFn>
GPU4DFT_HD inline void eval_rys_jk_quartet_tiered_uhf(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_PQ, bool do_j, bool do_ka, bool do_kb, double w_J, double w_K,
    const double* __restrict__ D_tot_cart, const double* __restrict__ D_alpha_cart,
    const double* __restrict__ D_beta_cart, int ncart_tot, int off_a, int off_b, int off_c,
    int off_d, AccumJBraFn accum_j_bra, AccumJKetFn accum_j_ket, AccumKaFn accum_ka,
    AccumKbFn accum_kb, double omega = 0.0) {
  if (la < 0 || la > MaxL || lb < 0 || lb > MaxL || lc < 0 || lc > MaxL || ld < 0 || ld > MaxL) {
    return;
  }

  constexpr int kMaxCart = (MaxL + 1) * (MaxL + 2) / 2;
  constexpr int kMaxCart2 = kMaxCart * kMaxCart;
  constexpr int kDim = MaxL + 1;
  constexpr int kMaxRoots = 2 * MaxL + 1;

  const int bra_begin = bra.prim_offset[ib], bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik], ket_end = ket.prim_offset[ik + 1];

  if (la == 0 && lb == 0 && lc == 0 && ld == 0) {
    double val = 0.0;
    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const R p = static_cast<R>(bra.exponent[ip]);
      const R Px = static_cast<R>(bra.Px[ip]);
      const R Py = static_cast<R>(bra.Py[ip]);
      const R Pz = static_cast<R>(bra.Pz[ip]);
      const R Cab = static_cast<R>(bra.prefactor[ip]);
      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const R q = static_cast<R>(ket.exponent[kp]);
        const R PQx = Px - static_cast<R>(ket.Px[kp]);
        const R PQy = Py - static_cast<R>(ket.Py[kp]);
        const R PQz = Pz - static_cast<R>(ket.Pz[kp]);
        const R Ccd = static_cast<R>(ket.prefactor[kp]);
        const R rho = p * q / (p + q);
        const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
        val += static_cast<double>(Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho) *
                                   boys_f0_erf(T, rho, static_cast<R>(omega)));
      }
    }
    if (do_j) {
      accum_j_bra(0, 0, w_J * val * D_tot_cart[off_c * ncart_tot + off_d]);
      if (!same_PQ) {
        accum_j_ket(off_c, off_d, w_J * val * D_tot_cart[off_a * ncart_tot + off_b]);
      }
    }
    const double wk = w_K * val;
    if (do_ka) {
      accum_ka(off_a, off_c, wk * D_alpha_cart[off_b * ncart_tot + off_d]);
      accum_ka(off_b, off_c, wk * D_alpha_cart[off_a * ncart_tot + off_d]);
      accum_ka(off_a, off_d, wk * D_alpha_cart[off_b * ncart_tot + off_c]);
      accum_ka(off_b, off_d, wk * D_alpha_cart[off_a * ncart_tot + off_c]);
    }
    if (do_kb) {
      accum_kb(off_a, off_c, wk * D_beta_cart[off_b * ncart_tot + off_d]);
      accum_kb(off_b, off_c, wk * D_beta_cart[off_a * ncart_tot + off_d]);
      accum_kb(off_a, off_d, wk * D_beta_cart[off_b * ncart_tot + off_c]);
      accum_kb(off_b, off_d, wk * D_beta_cart[off_a * ncart_tot + off_c]);
    }
    return;
  }

  const int nca = n_cart(la), ncb = n_cart(lb), ncc = n_cart(lc), ncd = n_cart(ld);
  const int n_roots = (la + lb + lc + ld) / 2 + 1;
  const double Ax = bra.Ax[ib], Ay = bra.Ay[ib], Az = bra.Az[ib];
  const double Bx = bra.Bx[ib], By = bra.By[ib], Bz = bra.Bz[ib];
  const double Cx = ket.Ax[ik], Cy = ket.Ay[ik], Cz = ket.Az[ik];
  const double Dx = ket.Bx[ik], Dy = ket.By[ik], Dz = ket.Bz[ik];
  const R ABx = static_cast<R>(Ax - Bx), ABy = static_cast<R>(Ay - By),
          ABz = static_cast<R>(Az - Bz);
  const R CDx = static_cast<R>(Cx - Dx), CDy = static_cast<R>(Cy - Dy),
          CDz = static_cast<R>(Cz - Dz);

  CartTriple ta[kMaxCart], tb[kMaxCart], tc[kMaxCart], td[kMaxCart];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib_c = 0; ib_c < ncb; ++ib_c) tb[ib_c] = cart_triple(lb, ib_c);
  for (int ic = 0; ic < ncc; ++ic) tc[ic] = cart_triple(lc, ic);
  for (int id = 0; id < ncd; ++id) td[id] = cart_triple(ld, id);

  if constexpr (MaxL == 1) {
    R eri_cart[81];
    const int n_comp = nca * ncb * ncc * ncd;
    for (int idx = 0; idx < n_comp; ++idx) eri_cart[idx] = R(0);

    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const R p = static_cast<R>(bra.exponent[ip]);
      const R Px = static_cast<R>(bra.Px[ip]);
      const R Py = static_cast<R>(bra.Py[ip]);
      const R Pz = static_cast<R>(bra.Pz[ip]);
      const R Cab = static_cast<R>(bra.prefactor[ip]);
      const R PAx = Px - static_cast<R>(Ax);
      const R PAy = Py - static_cast<R>(Ay);
      const R PAz = Pz - static_cast<R>(Az);
      const R inv_2p = R(0.5) / p;

      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const R q = static_cast<R>(ket.exponent[kp]);
        const R Qx = static_cast<R>(ket.Px[kp]);
        const R Qy = static_cast<R>(ket.Py[kp]);
        const R Qz = static_cast<R>(ket.Pz[kp]);
        const R Ccd = static_cast<R>(ket.prefactor[kp]);
        const R PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
        const R pq_sum = p + q;
        const R rho = p * q / pq_sum;
        const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
        const R scale = Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho);

        R u_roots[kMaxRoots];
        R w_weights[kMaxRoots];
        rys_roots_weights_erf<R>(n_roots, T, rho, static_cast<R>(omega), u_roots, w_weights);

        const R inv_2q = R(0.5) / q;
        const R inv_2pq = R(0.5) / pq_sum;
        const R rho_over_p = rho / p;
        const R rho_over_q = rho / q;
        const R QCx = Qx - static_cast<R>(Cx);
        const R QCy = Qy - static_cast<R>(Cy);
        const R QCz = Qz - static_cast<R>(Cz);

        for (int r = 0; r < n_roots; ++r) {
          const R u = u_roots[r];
          const R Wr = scale * w_weights[r];
          const R B00 = inv_2pq * u;
          const R B10 = inv_2p * (R(1) - rho_over_p * u);
          const R B01 = inv_2q * (R(1) - rho_over_q * u);

          R Ix[kDim][kDim][kDim][kDim], Iy[kDim][kDim][kDim][kDim], Iz[kDim][kDim][kDim][kDim];
          jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAx - rho_over_p * u * PQx,
                                    QCx + rho_over_q * u * PQx, B10, B01, B00, ABx, CDx, Ix);
          jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAy - rho_over_p * u * PQy,
                                    QCy + rho_over_q * u * PQy, B10, B01, B00, ABy, CDy, Iy);
          jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAz - rho_over_p * u * PQz,
                                    QCz + rho_over_q * u * PQz, B10, B01, B00, ABz, CDz, Iz);

          int out_idx = 0;
          for (int ia = 0; ia < nca; ++ia) {
            for (int ib_c = 0; ib_c < ncb; ++ib_c) {
              for (int ic = 0; ic < ncc; ++ic) {
                for (int id = 0; id < ncd; ++id) {
                  eri_cart[out_idx++] +=
                      Wr * Ix[ta[ia].lx][tb[ib_c].lx][tc[ic].lx][td[id].lx] *
                      Iy[ta[ia].ly][tb[ib_c].ly][tc[ic].ly][td[id].ly] *
                      Iz[ta[ia].lz][tb[ib_c].lz][tc[ic].lz][td[id].lz];
                }
              }
            }
          }
        }
      }
    }

    const int ncd_blk = ncc * ncd;
    if (do_j) {
      R d_cd_sym[kMaxCart2];
      for (int ic = 0; ic < ncc; ++ic) {
        const int rc = off_c + ic;
        for (int id = 0; id < ncd; ++id) {
          d_cd_sym[ic * ncd + id] = static_cast<R>(D_tot_cart[rc * ncart_tot + (off_d + id)]);
        }
      }
      R j_cd_loc[kMaxCart2] = {R(0)};
      for (int ia = 0; ia < nca; ++ia) {
        const int ra = off_a + ia;
        for (int ib_c = 0; ib_c < ncb; ++ib_c) {
          const int rb = off_b + ib_c;
          const R dab_sym = static_cast<R>(D_tot_cart[ra * ncart_tot + rb]);
          const R* e_ab = eri_cart + (ia * ncb + ib_c) * ncd_blk;
          R sum_jab = R(0);
          for (int icd = 0; icd < ncd_blk; ++icd) {
            const R v = e_ab[icd];
            sum_jab += v * d_cd_sym[icd];
            if (!same_PQ) j_cd_loc[icd] += v * dab_sym;
          }
          accum_j_bra(ia, ib_c, w_J * static_cast<double>(sum_jab));
        }
      }
      if (!same_PQ) {
        for (int ic = 0; ic < ncc; ++ic)
          for (int id = 0; id < ncd; ++id)
            accum_j_ket(off_c + ic, off_d + id, w_J * static_cast<double>(j_cd_loc[ic * ncd + id]));
      }
    }

    auto contract_k_sp = [&](const double* __restrict__ D_spin, auto&& accum_spin) {
      R k_ac[kMaxCart2] = {R(0)};
      R k_bc[kMaxCart2] = {R(0)};
      R k_ad[kMaxCart2] = {R(0)};
      R k_bd[kMaxCart2] = {R(0)};
      for (int ia = 0; ia < nca; ++ia) {
        const int ra = off_a + ia;
        for (int ib_c = 0; ib_c < ncb; ++ib_c) {
          const int rb = off_b + ib_c;
          const R* e_ab = eri_cart + (ia * ncb + ib_c) * ncd_blk;
          for (int ic = 0; ic < ncc; ++ic) {
            const int rc = off_c + ic;
            const R d_ac = static_cast<R>(D_spin[ra * ncart_tot + rc]);
            const R d_bc = static_cast<R>(D_spin[rb * ncart_tot + rc]);
            for (int id = 0; id < ncd; ++id) {
              const int rd = off_d + id;
              const R v = e_ab[ic * ncd + id];
              k_ac[ia * ncc + ic] += v * static_cast<R>(D_spin[rb * ncart_tot + rd]);
              k_bc[ib_c * ncc + ic] += v * static_cast<R>(D_spin[ra * ncart_tot + rd]);
              k_ad[ia * ncd + id] += v * d_bc;
              k_bd[ib_c * ncd + id] += v * d_ac;
            }
          }
        }
      }
      for (int ia = 0; ia < nca; ++ia)
        for (int ic = 0; ic < ncc; ++ic)
          accum_spin(off_a + ia, off_c + ic, w_K * static_cast<double>(k_ac[ia * ncc + ic]));
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        for (int ic = 0; ic < ncc; ++ic)
          accum_spin(off_b + ib_c, off_c + ic, w_K * static_cast<double>(k_bc[ib_c * ncc + ic]));
      for (int ia = 0; ia < nca; ++ia)
        for (int id = 0; id < ncd; ++id)
          accum_spin(off_a + ia, off_d + id, w_K * static_cast<double>(k_ad[ia * ncd + id]));
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        for (int id = 0; id < ncd; ++id)
          accum_spin(off_b + ib_c, off_d + id, w_K * static_cast<double>(k_bd[ib_c * ncd + id]));
    };

    if (do_ka) contract_k_sp(D_alpha_cart, accum_ka);
    if (do_kb) contract_k_sp(D_beta_cart, accum_kb);
    return;
  }

  R j_ab[kMaxCart2] = {R(0)};
  R j_cd[kMaxCart2] = {R(0)};
  R ka_ac[kMaxCart2] = {R(0)}, ka_bc[kMaxCart2] = {R(0)}, ka_ad[kMaxCart2] = {R(0)},
    ka_bd[kMaxCart2] = {R(0)};
  R kb_ac[kMaxCart2] = {R(0)}, kb_bc[kMaxCart2] = {R(0)}, kb_ad[kMaxCart2] = {R(0)},
    kb_bd[kMaxCart2] = {R(0)};

  for (int ip = bra_begin; ip < bra_end; ++ip) {
    const R p = static_cast<R>(bra.exponent[ip]);
    const R Px = static_cast<R>(bra.Px[ip]);
    const R Py = static_cast<R>(bra.Py[ip]);
    const R Pz = static_cast<R>(bra.Pz[ip]);
    const R Cab = static_cast<R>(bra.prefactor[ip]);
    const R PAx = Px - static_cast<R>(Ax);
    const R PAy = Py - static_cast<R>(Ay);
    const R PAz = Pz - static_cast<R>(Az);
    const R inv_2p = R(0.5) / p;

    for (int kp = ket_begin; kp < ket_end; ++kp) {
      const R q = static_cast<R>(ket.exponent[kp]);
      const R Qx = static_cast<R>(ket.Px[kp]);
      const R Qy = static_cast<R>(ket.Py[kp]);
      const R Qz = static_cast<R>(ket.Pz[kp]);
      const R Ccd = static_cast<R>(ket.prefactor[kp]);
      const R PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
      const R pq_sum = p + q;
      const R rho = p * q / pq_sum;
      const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
      const R scale = Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho);

      R u_roots[kMaxRoots];
      R w_weights[kMaxRoots];
      rys_roots_weights_erf<R>(n_roots, T, rho, static_cast<R>(omega), u_roots, w_weights);

      const R inv_2q = R(0.5) / q;
      const R inv_2pq = R(0.5) / pq_sum;
      const R rho_over_p = rho / p;
      const R rho_over_q = rho / q;
      const R QCx = Qx - static_cast<R>(Cx);
      const R QCy = Qy - static_cast<R>(Cy);
      const R QCz = Qz - static_cast<R>(Cz);

      for (int r = 0; r < n_roots; ++r) {
        const R u = u_roots[r];
        const R Wr = scale * w_weights[r];
        const R B00 = inv_2pq * u;
        const R B10 = inv_2p * (R(1) - rho_over_p * u);
        const R B01 = inv_2q * (R(1) - rho_over_q * u);

        R Ix[kDim][kDim][kDim][kDim], Iy[kDim][kDim][kDim][kDim], Iz[kDim][kDim][kDim][kDim];
        jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAx - rho_over_p * u * PQx,
                                  QCx + rho_over_q * u * PQx, B10, B01, B00, ABx, CDx, Ix);
        jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAy - rho_over_p * u * PQy,
                                  QCy + rho_over_q * u * PQy, B10, B01, B00, ABy, CDy, Iy);
        jk_rys_1d_tiered<MaxL, R>(la, lb, lc, ld, PAz - rho_over_p * u * PQz,
                                  QCz + rho_over_q * u * PQz, B10, B01, B00, ABz, CDz, Iz);

        for (int ia = 0; ia < nca; ++ia) {
          const int ra = off_a + ia;
          for (int ib_c = 0; ib_c < ncb; ++ib_c) {
            const int rb = off_b + ib_c;
            const R dab = static_cast<R>(D_tot_cart[ra * ncart_tot + rb]);
            R sum_jab = R(0);
            for (int ic = 0; ic < ncc; ++ic) {
              const int rc = off_c + ic;
              const R daac = do_ka ? static_cast<R>(D_alpha_cart[ra * ncart_tot + rc]) : R(0);
              const R dabc = do_ka ? static_cast<R>(D_alpha_cart[rb * ncart_tot + rc]) : R(0);
              const R dbac = do_kb ? static_cast<R>(D_beta_cart[ra * ncart_tot + rc]) : R(0);
              const R dbbc = do_kb ? static_cast<R>(D_beta_cart[rb * ncart_tot + rc]) : R(0);
              for (int id = 0; id < ncd; ++id) {
                const int rd = off_d + id;
                const R v = Wr * Ix[ta[ia].lx][tb[ib_c].lx][tc[ic].lx][td[id].lx] *
                            Iy[ta[ia].ly][tb[ib_c].ly][tc[ic].ly][td[id].ly] *
                            Iz[ta[ia].lz][tb[ib_c].lz][tc[ic].lz][td[id].lz];
                if (do_j) {
                  sum_jab += v * static_cast<R>(D_tot_cart[rc * ncart_tot + rd]);
                  if (!same_PQ) j_cd[ic * ncd + id] += v * dab;
                }
                if (do_ka) {
                  ka_ac[ia * ncc + ic] += v * static_cast<R>(D_alpha_cart[rb * ncart_tot + rd]);
                  ka_bc[ib_c * ncc + ic] += v * static_cast<R>(D_alpha_cart[ra * ncart_tot + rd]);
                  ka_ad[ia * ncd + id] += v * dabc;
                  ka_bd[ib_c * ncd + id] += v * daac;
                }
                if (do_kb) {
                  kb_ac[ia * ncc + ic] += v * static_cast<R>(D_beta_cart[rb * ncart_tot + rd]);
                  kb_bc[ib_c * ncc + ic] += v * static_cast<R>(D_beta_cart[ra * ncart_tot + rd]);
                  kb_ad[ia * ncd + id] += v * dbbc;
                  kb_bd[ib_c * ncd + id] += v * dbac;
                }
              }
            }
            if (do_j) j_ab[ia * ncb + ib_c] += sum_jab;
          }
        }
      }
    }
  }

  if (do_j) {
    for (int ia = 0; ia < nca; ++ia)
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        accum_j_bra(ia, ib_c, w_J * static_cast<double>(j_ab[ia * ncb + ib_c]));
    if (!same_PQ) {
      for (int ic = 0; ic < ncc; ++ic)
        for (int id = 0; id < ncd; ++id)
          accum_j_ket(off_c + ic, off_d + id, w_J * static_cast<double>(j_cd[ic * ncd + id]));
    }
  }
  if (do_ka) {
    for (int ia = 0; ia < nca; ++ia)
      for (int ic = 0; ic < ncc; ++ic)
        accum_ka(off_a + ia, off_c + ic, w_K * static_cast<double>(ka_ac[ia * ncc + ic]));
    for (int ib_c = 0; ib_c < ncb; ++ib_c)
      for (int ic = 0; ic < ncc; ++ic)
        accum_ka(off_b + ib_c, off_c + ic, w_K * static_cast<double>(ka_bc[ib_c * ncc + ic]));
    for (int ia = 0; ia < nca; ++ia)
      for (int id = 0; id < ncd; ++id)
        accum_ka(off_a + ia, off_d + id, w_K * static_cast<double>(ka_ad[ia * ncd + id]));
    for (int ib_c = 0; ib_c < ncb; ++ib_c)
      for (int id = 0; id < ncd; ++id)
        accum_ka(off_b + ib_c, off_d + id, w_K * static_cast<double>(ka_bd[ib_c * ncd + id]));
  }
  if (do_kb) {
    for (int ia = 0; ia < nca; ++ia)
      for (int ic = 0; ic < ncc; ++ic)
        accum_kb(off_a + ia, off_c + ic, w_K * static_cast<double>(kb_ac[ia * ncc + ic]));
    for (int ib_c = 0; ib_c < ncb; ++ib_c)
      for (int ic = 0; ic < ncc; ++ic)
        accum_kb(off_b + ib_c, off_c + ic, w_K * static_cast<double>(kb_bc[ib_c * ncc + ic]));
    for (int ia = 0; ia < nca; ++ia)
      for (int id = 0; id < ncd; ++id)
        accum_kb(off_a + ia, off_d + id, w_K * static_cast<double>(kb_ad[ia * ncd + id]));
    for (int ib_c = 0; ib_c < ncb; ++ib_c)
      for (int id = 0; id < ncd; ++id)
        accum_kb(off_b + ib_c, off_d + id, w_K * static_cast<double>(kb_bd[ib_c * ncd + id]));
  }
}

#if defined(GPU4DFT_HAVE_CUDA)
DirectJKCudaCache* create_direct_jk_cuda_cache(
    const Basis& basis, const std::vector<ShellPairData>& pair_lists,
    const std::vector<int>& shell_cart_offset, int ncart_total);
void destroy_direct_jk_cuda_cache(DirectJKCudaCache* p);

void run_direct_jk_cuda_device(DirectJKCudaCache& cache, const double* d_dm, double* d_vj,
                               double* d_vk, const JKOptions& opts, void* stream);
void run_direct_jk_cuda_host(DirectJKCudaCache& cache, const double* h_dm, double* h_vj,
                             double* h_vk, const JKOptions& opts);

void run_direct_jk_uhf_cuda_device(DirectJKCudaCache& cache, const double* d_dm_alpha,
                                   const double* d_dm_beta, double* d_vj_tot, double* d_vk_alpha,
                                   double* d_vk_beta, const JKOptions& opts, void* stream);
void run_direct_jk_uhf_cuda_host(DirectJKCudaCache& cache, const double* h_dm_alpha,
                                 const double* h_dm_beta, double* h_vj_tot, double* h_vk_alpha,
                                 double* h_vk_beta, const JKOptions& opts);

RIJKCudaCache* build_rijk_cuda_cache_native(
    const Basis& basis, const Basis& aux_basis, const std::vector<ShellPairData>& orb_lists,
    const std::vector<ShellPairData>& aux_lists, const std::vector<int>& orb_shell_cart_off,
    const std::vector<int>& aux_shell_cart_off, int ncart_orb, int ncart_aux,
    double rsh_omega = 0.0);
void destroy_rijk_cuda_cache(RIJKCudaCache* p);
void release_rijk_workspaces(RIJKCudaCache& cache);
bool download_rijk_metric_factor(const RIJKCudaCache& cache, double omega, bool& is_cholesky_out,
                                 std::vector<double>& factor_out);

void run_rijk_mo_cuda_device(RIJKCudaCache& cache, const double* d_dm, const double* d_c_occ,
                             int n_occ, double occ_scale, double* d_vj, double* d_vk,
                             const JKOptions& opts, void* stream);
void run_rijk_mo_cuda(RIJKCudaCache& cache, const double* h_dm, const double* h_c_occ, int n_occ,
                      double occ_scale, double* h_vj, double* h_vk, const JKOptions& opts);
void run_rijk_uks_mo_cuda_device(RIJKCudaCache& cache, const double* d_dm_alpha,
                                 const double* d_dm_beta, const double* d_c_occ_alpha,
                                 int n_alpha, const double* d_c_occ_beta, int n_beta,
                                 double* d_vj_tot, double* d_vk_alpha, double* d_vk_beta,
                                 const JKOptions& opts, void* stream);
void run_rijk_uks_mo_cuda(RIJKCudaCache& cache, const double* h_dm_alpha,
                          const double* h_dm_beta, const double* h_c_occ_alpha, int n_alpha,
                          const double* h_c_occ_beta, int n_beta, double* h_vj_tot,
                          double* h_vk_alpha, double* h_vk_beta, const JKOptions& opts);
void run_ri_jk_grad_cuda(RIJKCudaCache& cache, std::span<const Atom> atoms,
                         const double* h_dm_alpha, const double* h_c_occ_alpha, int n_alpha,
                         const double* h_dm_beta, const double* h_c_occ_beta, int n_beta,
                         bool unrestricted, double j_fraction, double k_sr, double k_lr,
                         double omega, std::vector<double>& grad_out);

void run_jk_nuclear_grad_cuda(const std::vector<ShellPairData>& pair_lists,
                              const std::vector<GradQuartetTask>& tasks_sr,
                              const std::vector<GradQuartetTask>& tasks_lr,
                              const std::vector<int>& shell_cart_offset,
                              const std::vector<int>& shell_atom, int ncart_total, int natom,
                              const std::vector<double>& d_cart_host, double j_fraction,
                              double k_sr, double k_lr_delta, double omega,
                              std::vector<double>& grad_out);

void run_jk_nuclear_grad_uhf_cuda(
    const std::vector<ShellPairData>& pair_lists, const std::vector<GradQuartetTask>& tasks_sr,
    const std::vector<GradQuartetTask>& tasks_lr, const std::vector<int>& shell_cart_offset,
    const std::vector<int>& shell_atom, int ncart_total, int natom,
    const std::vector<double>& d_tot_cart_host, const std::vector<double>& d_alpha_cart_host,
    const std::vector<double>& d_beta_cart_host, double j_fraction, double k_sr, double k_lr_delta,
    double omega, std::vector<double>& grad_out);
#endif

}  // namespace detail

}  // namespace gpu4dft
