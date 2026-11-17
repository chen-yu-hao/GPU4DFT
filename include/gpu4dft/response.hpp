#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/boys.hpp"
#include "gpu4dft/config.hpp"
#include "gpu4dft/device_pool.hpp"
#include "gpu4dft/harmonics.hpp"
#include "gpu4dft/linalg.hpp"
#include "gpu4dft/math.hpp"
#include "gpu4dft/one_electron.hpp"
#include "gpu4dft/rys.hpp"
#include "gpu4dft/shell_pairs.hpp"
#include "gpu4dft/types.hpp"
#include "gpu4dft/workflow.hpp"
#include "gpu4dft/xc.hpp"

namespace gpu4dft {

// ===========================================================================
// 1. AO Electric Dipole Integrals & Permanent Molecular Dipole Moment
// ===========================================================================

// Evaluates AO electric dipole integral matrices \mu^{(c)}_{uv} = -<u | r_c - O_c | v>
// for c = 0 (x), 1 (y), 2 (z), each row-major [nao * nao] in spherical AO order.
// Supports s, p, d, f, g shells (0 <= l <= 4) on Host (OpenMP) and CUDA GPU.
std::array<std::vector<double>, 3> dipole_matrices(
    const Basis& basis, std::array<double, 3> origin = {0.0, 0.0, 0.0},
    Backend backend = Backend::Host);

// Evaluates total permanent molecular dipole moment (in atomic units e * Bohr):
//   \mu_{tot} = \sum_A Z_A (R_A - O) + Tr(D * \mu_{AO}).
std::array<double, 3> molecular_dipole_moment(
    const Basis& basis, std::span<const Atom> atoms, const double* dm_host,
    std::array<double, 3> origin = {0.0, 0.0, 0.0}, Backend backend = Backend::Host);

// ===========================================================================
// 2. Closed-Shell Singlet Linear-Response Excited-State Solver
//    (CIS / Tamm-Dancoff Approximation TDA / Full TDDFT & TDHF Casida RPA)
// ===========================================================================

enum class ResponseMethod {
  TDA,    // Tamm-Dancoff Approximation (and CIS when exact_exchange_fraction = 1.0, no XC)
  TDDFT,  // Full Casida Linear-Response TDDFT / TDHF (Random Phase Approximation)
};

const char* to_string(ResponseMethod method);

struct ResponseOptions {
  Backend backend = Backend::Cuda;
  ResponseMethod method = ResponseMethod::TDDFT;

  // GEMM execution mode for GPU/Host AO <-> MO transition density transforms and contractions:
  // FP64, FP32 Tensor Core, or OzakiInt8 (INT8 Tensor Core mantissa slicing).
  GemmMode gemm_mode = GemmMode::FP64;
  bool dynamic_precision = false;          // Optional FP32/Ozaki early Davidson -> FP64 polish
  GemmMode early_gemm_mode = GemmMode::FP32;
  double fp64_switch_residual = 1e-3;
  int ozaki_slices = 8;

  int n_states = 5;                        // Number of lowest singlet excited states
  int max_iter = 60;                       // Maximum Block-Davidson iterations
  int max_subspace = 100;                  // Maximum Krylov subspace vectors before restart
  double conv_tol = 1e-7;                  // Max residual L2 norm threshold (Eh accuracy ~ tol^2)
  double exact_exchange_fraction = 1.0;    // 1.0 = CIS/TDHF, 0.0 = pure DFT, 0.20 = B3LYP, 0.25 = PBE0
  double denom_floor = 1e-8;               // Davidson diagonal preconditioner safeguard
};

// Decoupled AO linear-response callbacks supporting BOTH GPU device pointers
// (`device_resident = true`, `fxc_device_resident = true`) and Host buffers,
// mirroring ScfDriver's FockCallbacks so LinearResponseDriver works standalone
// in tests AND wires directly to DirectJKEngine, RIJKEngine, and XcEngine::eval_fxc_device.
struct ResponseCallbacks {
  // Contracts a (generally NON-SYMMETRIC) AO transition density matrix `dm_ptr` [nao x nao]:
  //   J(dm)_{uv} = \sum_{ls} (uv|ls) dm_{ls}    (written if `vj_out_ptr != nullptr`)
  //   K(dm)_{uv} = \sum_{ls} (ul|vs) dm_{ls}    (written if `vk_out_ptr != nullptr`, NOT symmetrized!)
  std::function<void(const double* dm_ptr, double* vj_out_ptr, double* vk_out_ptr, void* stream)>
      build_jk;

  // Optional XC Hessian contraction for a SYMMETRIC perturbation density `sym_dm_ptr` [nao x nao]:
  //   F_xc[sym_dm]_{uv} = d/d\lambda V_xc^{RKS}[D_0 + \lambda sym_dm]_{uv} |_{\lambda=0}
  // Leave empty for CIS / TDHF.
  std::function<void(const double* sym_dm_ptr, double* fxc_out_ptr, void* stream)> build_fxc;

  // When true on Backend::Cuda, pointers passed to `build_jk` are GPU device pointers.
  bool device_resident = false;

  // When true on Backend::Cuda, pointers passed to `build_fxc` are GPU device pointers.
  bool fxc_device_resident = false;
};

struct DavidsonIterationLog {
  int iteration = 0;
  int subspace_dim = 0;
  double max_residual = 0.0;
  GemmMode gemm_mode = GemmMode::FP64;
};

struct ExcitedState {
  int state_index = 0;
  double energy_hartree = 0.0;
  double energy_ev = 0.0;
  std::array<double, 3> transition_dipole{};  // d_k = sqrt(2) Tr[\mu C_o (X_k + Y_k) C_v^T] (a.u.)
  double oscillator_strength = 0.0;           // f_k = (2/3) \omega_k |d_k|^2 (dipole length gauge)
  double residual_norm = 0.0;
  int dominant_occ = 0;
  int dominant_vir = 0;
  double dominant_weight = 0.0;               // |X_{ia}|^2 - |Y_{ia}|^2
};

struct ResponseResult {
  bool converged = false;
  int iterations = 0;
  int n_states = 0;
  int nao = 0;
  int nmo = 0;
  int n_occ = 0;
  int n_vir = 0;
  int matvec_count = 0;
  double wall_time_s = 0.0;

  std::vector<double> excitation_energies;               // [n_states] Hartree
  std::vector<double> excitation_energies_ev;            // [n_states] eV
  std::vector<double> oscillator_strengths;              // [n_states] dimensionless f_osc
  std::vector<std::array<double, 3>> transition_dipoles; // [n_states] (dx, dy, dz) in a.u.
  std::vector<double> x_vectors;                         // [n_states * n_occ * n_vir] spin-adapted X
  std::vector<double> y_vectors;                         // [n_states * n_occ * n_vir] spin-adapted Y
  std::vector<ExcitedState> states;
  std::vector<DavidsonIterationLog> history;
};

class LinearResponseDriver {
 public:
  explicit LinearResponseDriver(ResponseOptions opts = {}, DeviceMemoryPool* pool = nullptr);
  ~LinearResponseDriver();

  LinearResponseDriver(const LinearResponseDriver&) = delete;
  LinearResponseDriver& operator=(const LinearResponseDriver&) = delete;
  LinearResponseDriver(LinearResponseDriver&&) noexcept;
  LinearResponseDriver& operator=(LinearResponseDriver&&) noexcept;

  const ResponseOptions& options() const noexcept;
  void set_options(ResponseOptions opts);

  // Solves closed-shell singlet CIS / TDA / TDHF / TDDFT excited states from canonical MOs
  // `mo_energies_host` [nmo] and `mo_coeff_host` [nao x nmo] (row-major, C[mu, p] = mo_coeff[mu*nmo + p]).
  // If `ao_dipoles_host` (3 x [nao*nao]) is supplied, computes transition dipoles and oscillator strengths.
  ResponseResult run(int nao, int nmo, int n_occ, const double* mo_energies_host,
                     const double* mo_coeff_host, const ResponseCallbacks& callbacks,
                     const std::array<std::vector<double>, 3>* ao_dipoles_host = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Creates GPU-resident (on Backend::Cuda) or Host (on Backend::Host) non-symmetric J/K
// ResponseCallbacks from an in-core 4-index ERI tensor `h_eri` [nao, nao, nao, nao].
ResponseCallbacks make_in_core_eri_response_callbacks(
    int nao, const double* h_eri, Backend backend = Backend::Cuda,
    bool need_k = true, GemmMode gemm_mode = GemmMode::FP64, int ozaki_slices = 8);

// Creates Direct on-the-fly shell-pair Rys-quadrature non-symmetric J/K ResponseCallbacks
// on Host (OpenMP) or CUDA GPU without O(nao^4) ERI tensor allocation.
ResponseCallbacks make_direct_jk_response_callbacks(
    const Basis& basis, Backend backend = Backend::Cuda, bool need_k = true,
    double schwarz_cutoff = 1e-12, bool force_on_the_fly_host = false);

// Attaches GPU-resident (on Backend::Cuda) or Host (on Backend::Host) 2nd-derivative XC
// kernel contraction `F_xc[sym_dm]` (using generated symbolic `xc_kernels.hpp` for LDA, GGA,
// Hybrid, and Meta-GGA functionals) onto `callbacks` around ground-state density `dm0_host`.
void attach_fxc_response_callback(
    ResponseCallbacks& callbacks, const Basis& basis, const MolecularGrid& grid,
    std::string_view xc_name, const double* dm0_host, Backend backend = Backend::Cuda,
    double rho_cutoff = 1e-15);

// End-to-end molecular CIS / TDA / TDHF / TDDFT convenience driver from converged ScfResult.
ResponseResult run_linear_response(
    const Basis& basis, std::span<const Atom> atoms, int n_occ, const ScfResult& scf_res,
    std::string_view xc_name = "hf", const ResponseOptions& opts = {},
    std::span<const double> external_grid_coords = {},
    std::span<const double> external_grid_weights = {},
    const GridOptions& grid_opts = {});

// ===========================================================================
// 3. Coupled-Perturbed HF / KS (CPHF / Z-Vector) & Static Polarizability
// ===========================================================================

struct CphfOptions {
  Backend backend = Backend::Cuda;
  GemmMode gemm_mode = GemmMode::FP64;
  int ozaki_slices = 8;
  int max_iter = 80;
  double conv_tol = 1e-9;
  double exact_exchange_fraction = 1.0;
  bool symmetric_perturbation = true;  // true: (A + B) Z = R (Z-vector / polarizability); false: (A - B) Z = R
};

struct CphfResult {
  bool converged = false;
  int iterations = 0;
  double residual_norm = 0.0;
  std::vector<double> z_mo;  // [n_occ * n_vir]
  std::vector<double> z_ao;  // [nao * nao] C_o Z C_v^T +/- C_v Z^T C_o^T
};

// Preconditioned Conjugate-Gradient / Krylov CPHF & CPKS linear-response solver.
CphfResult solve_cphf(int nao, int nmo, int n_occ, const double* mo_energies_host,
                      const double* mo_coeff_host, const double* rhs_mo_host,
                      const ResponseCallbacks& callbacks, const CphfOptions& opts = {});

// Computes the 3x3 static electric dipole polarizability tensor \alpha_{ab} (in Bohr^3)
// via CPHF / CPKS linear response.
std::array<double, 9> compute_static_polarizability(
    int nao, int nmo, int n_occ, const double* mo_energies_host, const double* mo_coeff_host,
    const std::array<std::vector<double>, 3>& ao_dipoles_host, const ResponseCallbacks& callbacks,
    const CphfOptions& opts = {});

// ===========================================================================
// 4. Molecular Nuclear Hessian & Harmonic Vibrational Frequency Engine
// ===========================================================================

struct HessianOptions {
  Backend backend = Backend::Cuda;
  double step_bohr = 1e-3;       // Central 2-point finite-difference step \delta (Bohr)
  bool project_trans_rot = true; // Eckart translation + rotation projection
  double tr_svd_tol = 1e-6;      // Rank threshold for TR generator orthonormalization
  bool fixed_xc_grid = true;     // Hold XC quadrature grid fixed at equilibrium R_0 (matches PySCF grid_response=False)
};

struct VibrationalResult {
  int natom = 0;
  int n_vib = 0;
  int n_trans_rot = 0;
  int imaginary_count = 0;
  double zero_point_energy_hartree = 0.0;
  double wall_time_s = 0.0;

  std::vector<double> hessian;             // [3*natom * 3*natom] symmetrized Eh / Bohr^2
  std::vector<double> frequencies_cm1;     // [n_vib] cm^{-1} (negative for imaginary modes)
  std::vector<double> force_constants_au;  // [n_vib] eigenvalues in Eh / (Bohr^2 * amu)
  std::vector<double> reduced_masses_amu;  // [n_vib] reduced masses in amu
  std::vector<double> normal_modes;        // [n_vib * 3 * natom] normalized Cartesian displacements
  std::vector<double> mass_weighted_modes; // [n_vib * 3 * natom] orthonormal mass-weighted modes
};

// Returns standard atomic weight (amu) matching PySCF `pyscf.data.elements.MASSES` (Z = 1..86).
double standard_atomic_mass_amu(int atomic_number);

// Performs Eckart translational/rotational projection and mass-weighted diagonalization
// (on GPU via cuSOLVER or Host) of an existing [3*natom x 3*natom] Cartesian Hessian (Eh / Bohr^2).
VibrationalResult analyze_hessian(std::span<const Atom> atoms, const double* hessian_3n_3n_host,
                                  std::span<const double> custom_masses_amu = {},
                                  const HessianOptions& opts = {});

// Computes molecular nuclear Hessian via central 2-point seminumerical differentiation
// of analytical nuclear gradients `grad_fn(displaced_atoms)` [natom * 3] (Eh / Bohr),
// followed by Eckart projection and harmonic vibrational analysis.
VibrationalResult compute_nuclear_hessian(
    std::span<const Atom> atoms,
    const std::function<std::vector<double>(std::span<const Atom>)>& grad_fn,
    std::span<const double> custom_masses_amu = {}, const HessianOptions& opts = {});

// End-to-end molecular nuclear Hessian & harmonic vibrational frequency workflow
// driving GPU4DFT's analytical nuclear gradients (RHF / RKS / Hybrid DFT).
VibrationalResult compute_molecular_hessian(
    const Basis& basis, std::span<const Atom> atoms, int n_elec,
    WorkflowOptions wf_opts = {}, std::span<const double> custom_masses_amu = {},
    const HessianOptions& hess_opts = {},
    std::span<const double> external_grid_coords = {},
    std::span<const double> external_grid_weights = {});

// ===========================================================================
// Internal Host/Device Shared Detail Primitives
// ===========================================================================

namespace detail {

// Evaluates AO electric dipole shell-pair block \mu^{(c)}_{uv} = -<u | r_c - O_c | v>
// for c = 0, 1, 2 into row-major [nao x nao] matrices `mx, my, mz`.
GPU4DFT_HD inline void eval_dipole_pair(const BasisDeviceView& b, int ish, int jsh,
                                        double ox, double oy, double oz,
                                        double* mx, double* my, double* mz) {
  const int la = b.shell_l[ish];
  const int lb = b.shell_l[jsh];
  const int nca = n_cart(la);
  const int ncb = n_cart(lb);
  const int nsa = n_sph(la);
  const int nsb = n_sph(lb);

  const double Ax = b.shell_center[3 * ish + 0];
  const double Ay = b.shell_center[3 * ish + 1];
  const double Az = b.shell_center[3 * ish + 2];
  const double Bx = b.shell_center[3 * jsh + 0];
  const double By = b.shell_center[3 * jsh + 1];
  const double Bz = b.shell_center[3 * jsh + 2];
  const double ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;
  const double AB2 = ABx * ABx + ABy * ABy + ABz * ABz;
  const double BOx = Bx - ox, BOy = By - oy, BOz = Bz - oz;

  CartTriple ta[15], tb[15];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib = 0; ib < ncb; ++ib) tb[ib] = cart_triple(lb, ib);

  double cart_x[15 * 15] = {0.0};
  double cart_y[15 * 15] = {0.0};
  double cart_z[15 * 15] = {0.0};

  const int pa_beg = b.prim_offset[ish], pa_end = b.prim_offset[ish + 1];
  const int pb_beg = b.prim_offset[jsh], pb_end = b.prim_offset[jsh + 1];

  for (int ka = pa_beg; ka < pa_end; ++ka) {
    const double a = b.prim_exp[ka];
    const double ca = b.prim_coeff[ka];
    for (int kb = pb_beg; kb < pb_end; ++kb) {
      const double beta = b.prim_exp[kb];
      const double cb = b.prim_coeff[kb];
      const double p = a + beta;
      const double inv_p = 1.0 / p;
      const double inv_2p = 0.5 * inv_p;
      const double Px = (a * Ax + beta * Bx) * inv_p;
      const double Py = (a * Ay + beta * By) * inv_p;
      const double Pz = (a * Az + beta * Bz) * inv_p;
      const double ratio = kOneElecPi * inv_p;
      const double pref = ca * cb * harmonics::kY00 * harmonics::kY00 *
                          math::exp(-a * beta * inv_p * AB2) * ratio * math::sqrt(ratio);

      double Sx[7][8], Sy[7][8], Sz[7][8];
      one_elec_overlap_1d(la, lb + 1, inv_2p, Px - Ax, ABx, Sx);
      one_elec_overlap_1d(la, lb + 1, inv_2p, Py - Ay, ABy, Sy);
      one_elec_overlap_1d(la, lb + 1, inv_2p, Pz - Az, ABz, Sz);

      for (int ia = 0; ia < nca; ++ia) {
        const int ix = ta[ia].lx, iy = ta[ia].ly, iz = ta[ia].lz;
        for (int ib = 0; ib < ncb; ++ib) {
          const int jx = tb[ib].lx, jy = tb[ib].ly, jz = tb[ib].lz;
          const double sx = Sx[ix][jx], sy = Sy[iy][jy], sz = Sz[iz][jz];
          const double dx = Sx[ix][jx + 1] + BOx * sx;
          const double dy = Sy[iy][jy + 1] + BOy * sy;
          const double dz = Sz[iz][jz + 1] + BOz * sz;
          // Electric dipole operator is -(\mathbf{r} - \mathbf{O})
          cart_x[ia * ncb + ib] -= pref * dx * sy * sz;
          cart_y[ia * ncb + ib] -= pref * sx * dy * sz;
          cart_z[ia * ncb + ib] -= pref * sx * sy * dz;
        }
      }
    }
  }

  double sph_x[9 * 9], sph_y[9 * 9], sph_z[9 * 9];
  harmonics::cart_to_sph_2d<double>(la, lb, cart_x, sph_x);
  harmonics::cart_to_sph_2d<double>(la, lb, cart_y, sph_y);
  harmonics::cart_to_sph_2d<double>(la, lb, cart_z, sph_z);

  const int i0 = b.shell_ao[ish];
  const int j0 = b.shell_ao[jsh];
  const int nao = b.nao;
  for (int ma = 0; ma < nsa; ++ma) {
    for (int mb = 0; mb < nsb; ++mb) {
      const int idx = ma * nsb + mb;
      const int uv = (i0 + ma) * nao + (j0 + mb);
      const int vu = (j0 + mb) * nao + (i0 + ma);
      mx[uv] = sph_x[idx];
      mx[vu] = sph_x[idx];
      my[uv] = sph_y[idx];
      my[vu] = sph_y[idx];
      mz[uv] = sph_z[idx];
      mz[vu] = sph_z[idx];
    }
  }
}

struct RespJKTaskTile {
  int bp = 0;
  int P = 0;
  int kp = 0;
  int Q_begin = 0;
  int Q_end = 0;
};

template <int MaxL = 4>
GPU4DFT_HD inline void resp_rys_1d(int la, int lb, int lc, int ld, double C00, double D00,
                                   double B10, double B01, double B00, double AB, double CD,
                                   double I[MaxL + 1][MaxL + 1][MaxL + 1][MaxL + 1]) {
  constexpr int kMaxSum = 2 * MaxL + 1;
  const int lab = la + lb;
  const int lcd = lc + ld;
  double G[kMaxSum][kMaxSum];
  G[0][0] = 1.0;
  for (int n = 0; n < lab; ++n) {
    G[n + 1][0] = C00 * G[n][0] + (n > 0 ? double(n) * B10 * G[n - 1][0] : 0.0);
  }
  for (int m = 0; m < lcd; ++m) {
    for (int n = 0; n <= lab; ++n) {
      double val = D00 * G[n][m];
      if (m > 0) val += double(m) * B01 * G[n][m - 1];
      if (n > 0) val += double(n) * B00 * G[n - 1][m];
      G[n][m + 1] = val;
    }
  }

  double H[MaxL + 1][MaxL + 1][kMaxSum];
  double cur[kMaxSum][kMaxSum];
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
      double kcur[kMaxSum];
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

// Exact 8-branch ordered shell-quartet contraction for NON-SYMMETRIC transition density `D_cart`:
//   J[u, v] += \sum_{l,s} (uv|ls) D_cart[l, s]
//   K[u, l] += \sum_{v,s} (uv|ls) D_cart[v, s]
// Covers every ordered 4-tuple generated by unordered shell pairs P = {ish, jsh} and Q = {ksh, lsh}
// with exact multiplicity 1. Templated on MaxL (`2` for s/p/d fast register path, `4` for f/g).
template <int MaxL = 4, typename AccumJFn, typename AccumKFn>
GPU4DFT_HD inline void eval_nonsym_jk_quartet(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_ij, bool same_kl, bool same_PQ, bool do_j, bool do_k, const double* D_cart,
    int ncart_tot, int off_a, int off_b, int off_c, int off_d, AccumJFn accum_j, AccumKFn accum_k) {
  constexpr double kTwoOverSqrtPi = 1.12837916709551257390;
  constexpr int kMaxCart = (MaxL + 1) * (MaxL + 2) / 2;
  constexpr int kMaxCart2 = kMaxCart * kMaxCart;
  const int nca = n_cart(la), ncb = n_cart(lb), ncc = n_cart(lc), ncd = n_cart(ld);

  const int bra_begin = bra.prim_offset[ib], bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik], ket_end = ket.prim_offset[ik + 1];

  if (la == 0 && lb == 0 && lc == 0 && ld == 0) {
    double v = 0.0;
    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const double p = bra.exponent[ip];
      const double Px = bra.Px[ip], Py = bra.Py[ip], Pz = bra.Pz[ip];
      const double Cab = bra.prefactor[ip];
      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const double q = ket.exponent[kp];
        const double PQx = Px - ket.Px[kp], PQy = Py - ket.Py[kp], PQz = Pz - ket.Pz[kp];
        const double Ccd = ket.prefactor[kp];
        const double rho = p * q / (p + q);
        const double T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
        v += Cab * Ccd * kTwoOverSqrtPi * math::sqrt(rho) * boys_f0(T);
      }
    }
    if (do_j) {
      const double d_cd = D_cart[off_c * ncart_tot + off_d] +
                          (same_kl ? 0.0 : D_cart[off_d * ncart_tot + off_c]);
      accum_j(off_a, off_b, v * d_cd);
      if (!same_ij) accum_j(off_b, off_a, v * d_cd);
      if (!same_PQ) {
        const double d_ab = D_cart[off_a * ncart_tot + off_b] +
                            (same_ij ? 0.0 : D_cart[off_b * ncart_tot + off_a]);
        accum_j(off_c, off_d, v * d_ab);
        if (!same_kl) accum_j(off_d, off_c, v * d_ab);
      }
    }
    if (do_k) {
      accum_k(off_a, off_c, v * D_cart[off_b * ncart_tot + off_d]);
      if (!same_ij) accum_k(off_b, off_c, v * D_cart[off_a * ncart_tot + off_d]);
      if (!same_kl) accum_k(off_a, off_d, v * D_cart[off_b * ncart_tot + off_c]);
      if (!same_ij && !same_kl) accum_k(off_b, off_d, v * D_cart[off_a * ncart_tot + off_c]);
      if (!same_PQ) {
        accum_k(off_c, off_a, v * D_cart[off_d * ncart_tot + off_b]);
        if (!same_kl) accum_k(off_d, off_a, v * D_cart[off_c * ncart_tot + off_b]);
        if (!same_ij) accum_k(off_c, off_b, v * D_cart[off_d * ncart_tot + off_a]);
        if (!same_kl && !same_ij) accum_k(off_d, off_b, v * D_cart[off_c * ncart_tot + off_a]);
      }
    }
    return;
  }

  double j_ab[kMaxCart2] = {0.0};
  double j_cd[kMaxCart2] = {0.0};
  double k_ac[kMaxCart2] = {0.0}, k_bc[kMaxCart2] = {0.0}, k_ad[kMaxCart2] = {0.0}, k_bd[kMaxCart2] = {0.0};
  double k_ca[kMaxCart2] = {0.0}, k_da[kMaxCart2] = {0.0}, k_cb[kMaxCart2] = {0.0}, k_db[kMaxCart2] = {0.0};

  const int n_roots = (la + lb + lc + ld) / 2 + 1;
  const double Ax = bra.Ax[ib], Ay = bra.Ay[ib], Az = bra.Az[ib];
  const double Bx = bra.Bx[ib], By = bra.By[ib], Bz = bra.Bz[ib];
  const double Cx = ket.Ax[ik], Cy = ket.Ay[ik], Cz = ket.Az[ik];
  const double Dx = ket.Bx[ik], Dy = ket.By[ik], Dz = ket.Bz[ik];
  const double ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;
  const double CDx = Cx - Dx, CDy = Cy - Dy, CDz = Cz - Dz;

  CartTriple ta[kMaxCart], tb[kMaxCart], tc[kMaxCart], td[kMaxCart];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib_c = 0; ib_c < ncb; ++ib_c) tb[ib_c] = cart_triple(lb, ib_c);
  for (int ic = 0; ic < ncc; ++ic) tc[ic] = cart_triple(lc, ic);
  for (int id = 0; id < ncd; ++id) td[id] = cart_triple(ld, id);

  for (int ip = bra_begin; ip < bra_end; ++ip) {
    const double p = bra.exponent[ip];
    const double Px = bra.Px[ip], Py = bra.Py[ip], Pz = bra.Pz[ip];
    const double Cab = bra.prefactor[ip];
    const double PAx = Px - Ax, PAy = Py - Ay, PAz = Pz - Az;
    const double inv_2p = 0.5 / p;

    for (int kp = ket_begin; kp < ket_end; ++kp) {
      const double q = ket.exponent[kp];
      const double Qx = ket.Px[kp], Qy = ket.Py[kp], Qz = ket.Pz[kp];
      const double Ccd = ket.prefactor[kp];
      const double PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
      const double pq_sum = p + q;
      const double rho = p * q / pq_sum;
      const double T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
      const double scale = Cab * Ccd * kTwoOverSqrtPi * math::sqrt(rho);

      double u_roots[9], w_weights[9];
      rys_roots_weights<double>(n_roots, T, u_roots, w_weights);

      const double inv_2q = 0.5 / q;
      const double inv_2pq = 0.5 / pq_sum;
      const double rho_over_p = rho / p;
      const double rho_over_q = rho / q;
      const double QCx = Qx - Cx, QCy = Qy - Cy, QCz = Qz - Cz;

      for (int r = 0; r < n_roots; ++r) {
        const double u = u_roots[r];
        const double Wr = scale * w_weights[r];
        const double B00 = inv_2pq * u;
        const double B10 = inv_2p * (1.0 - rho_over_p * u);
        const double B01 = inv_2q * (1.0 - rho_over_q * u);

        double Ix[MaxL + 1][MaxL + 1][MaxL + 1][MaxL + 1];
        double Iy[MaxL + 1][MaxL + 1][MaxL + 1][MaxL + 1];
        double Iz[MaxL + 1][MaxL + 1][MaxL + 1][MaxL + 1];
        resp_rys_1d<MaxL>(la, lb, lc, ld, PAx - rho_over_p * u * PQx, QCx + rho_over_q * u * PQx,
                          B10, B01, B00, ABx, CDx, Ix);
        resp_rys_1d<MaxL>(la, lb, lc, ld, PAy - rho_over_p * u * PQy, QCy + rho_over_q * u * PQy,
                          B10, B01, B00, ABy, CDy, Iy);
        resp_rys_1d<MaxL>(la, lb, lc, ld, PAz - rho_over_p * u * PQz, QCz + rho_over_q * u * PQz,
                          B10, B01, B00, ABz, CDz, Iz);

        for (int ia = 0; ia < nca; ++ia) {
          const int a_idx = off_a + ia;
          for (int ib_c = 0; ib_c < ncb; ++ib_c) {
            const int b_idx = off_b + ib_c;
            const double d_ab_sym =
                D_cart[a_idx * ncart_tot + b_idx] +
                (same_ij ? 0.0 : D_cart[b_idx * ncart_tot + a_idx]);
            double sum_jab = 0.0;
            for (int ic = 0; ic < ncc; ++ic) {
              const int c_idx = off_c + ic;
              for (int id = 0; id < ncd; ++id) {
                const int d_idx = off_d + id;
                const double v = Wr * Ix[ta[ia].lx][tb[ib_c].lx][tc[ic].lx][td[id].lx] *
                                 Iy[ta[ia].ly][tb[ib_c].ly][tc[ic].ly][td[id].ly] *
                                 Iz[ta[ia].lz][tb[ib_c].lz][tc[ic].lz][td[id].lz];
                if (do_j) {
                  const double d_cd_sym =
                      D_cart[c_idx * ncart_tot + d_idx] +
                      (same_kl ? 0.0 : D_cart[d_idx * ncart_tot + c_idx]);
                  sum_jab += v * d_cd_sym;
                  if (!same_PQ) j_cd[ic * ncd + id] += v * d_ab_sym;
                }
                if (do_k) {
                  k_ac[ia * ncc + ic] += v * D_cart[b_idx * ncart_tot + d_idx];
                  if (!same_ij) k_bc[ib_c * ncc + ic] += v * D_cart[a_idx * ncart_tot + d_idx];
                  if (!same_kl) k_ad[ia * ncd + id] += v * D_cart[b_idx * ncart_tot + c_idx];
                  if (!same_ij && !same_kl)
                    k_bd[ib_c * ncd + id] += v * D_cart[a_idx * ncart_tot + c_idx];
                  if (!same_PQ) {
                    k_ca[ic * nca + ia] += v * D_cart[d_idx * ncart_tot + b_idx];
                    if (!same_kl) k_da[id * nca + ia] += v * D_cart[c_idx * ncart_tot + b_idx];
                    if (!same_ij) k_cb[ic * ncb + ib_c] += v * D_cart[d_idx * ncart_tot + a_idx];
                    if (!same_kl && !same_ij)
                      k_db[id * ncb + ib_c] += v * D_cart[c_idx * ncart_tot + a_idx];
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
    for (int ia = 0; ia < nca; ++ia) {
      for (int ib_c = 0; ib_c < ncb; ++ib_c) {
        const double val = j_ab[ia * ncb + ib_c];
        accum_j(off_a + ia, off_b + ib_c, val);
        if (!same_ij) accum_j(off_b + ib_c, off_a + ia, val);
      }
    }
    if (!same_PQ) {
      for (int ic = 0; ic < ncc; ++ic) {
        for (int id = 0; id < ncd; ++id) {
          const double val = j_cd[ic * ncd + id];
          accum_j(off_c + ic, off_d + id, val);
          if (!same_kl) accum_j(off_d + id, off_c + ic, val);
        }
      }
    }
  }

  if (do_k) {
    for (int ia = 0; ia < nca; ++ia)
      for (int ic = 0; ic < ncc; ++ic) accum_k(off_a + ia, off_c + ic, k_ac[ia * ncc + ic]);
    if (!same_ij) {
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        for (int ic = 0; ic < ncc; ++ic) accum_k(off_b + ib_c, off_c + ic, k_bc[ib_c * ncc + ic]);
    }
    if (!same_kl) {
      for (int ia = 0; ia < nca; ++ia)
        for (int id = 0; id < ncd; ++id) accum_k(off_a + ia, off_d + id, k_ad[ia * ncd + id]);
    }
    if (!same_ij && !same_kl) {
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        for (int id = 0; id < ncd; ++id) accum_k(off_b + ib_c, off_d + id, k_bd[ib_c * ncd + id]);
    }
    if (!same_PQ) {
      for (int ic = 0; ic < ncc; ++ic)
        for (int ia = 0; ia < nca; ++ia) accum_k(off_c + ic, off_a + ia, k_ca[ic * nca + ia]);
      if (!same_kl) {
        for (int id = 0; id < ncd; ++id)
          for (int ia = 0; ia < nca; ++ia) accum_k(off_d + id, off_a + ia, k_da[id * nca + ia]);
      }
      if (!same_ij) {
        for (int ic = 0; ic < ncc; ++ic)
          for (int ib_c = 0; ib_c < ncb; ++ib_c)
            accum_k(off_c + ic, off_b + ib_c, k_cb[ic * ncb + ib_c]);
      }
      if (!same_kl && !same_ij) {
        for (int id = 0; id < ncd; ++id)
          for (int ib_c = 0; ib_c < ncb; ++ib_c)
            accum_k(off_d + id, off_b + ib_c, k_db[id * ncb + ib_c]);
      }
    }
  }
}

#if defined(GPU4DFT_HAVE_CUDA)
void run_dipole_matrices_cuda(const BasisDeviceView& host_view, std::array<double, 3> origin,
                              std::array<std::vector<double>, 3>& out);

struct CudaInCoreResponseJK;
std::shared_ptr<CudaInCoreResponseJK> create_cuda_incore_response_jk(
    int nao, const double* h_eri, bool need_k);
void eval_cuda_incore_response_jk(CudaInCoreResponseJK& state, const double* d_dm,
                                  double* d_vj, double* d_vk, GemmMode mode,
                                  int ozaki_slices, StreamHandle stream);

struct CudaDirectResponseJK;
std::shared_ptr<CudaDirectResponseJK> create_cuda_direct_response_jk(
    const Basis& basis, bool need_k, double schwarz_cutoff);
void eval_cuda_direct_response_jk(CudaDirectResponseJK& state, const double* d_dm,
                                  double* d_vj, double* d_vk, StreamHandle stream);

struct CudaFxcEngine;
std::shared_ptr<CudaFxcEngine> create_cuda_fxc_engine(
    const FlatBasisData& flat_basis, const MolecularGrid& grid, int xc_func_id, int xc_family,
    const double* dm0_host, double rho_cutoff);
void eval_cuda_fxc_device(CudaFxcEngine& engine, const double* d_sym_dm, double* d_fxc_out,
                          StreamHandle stream);

// Device-side elementwise & matrix helper kernels for GPU-resident Davidson / CPHF:
// Forms d_sym = d_M + d_M^T [nao x nao].
void form_sym_sum_transpose_device(int nao, const double* d_M, double* d_sym,
                                   StreamHandle stream = nullptr);

// Assembles TDA AO effective potential:
//   d_V[u, v] = 2 * d_J[u, v] + (has_fxc ? d_Fxc[u, v] : 0) - alpha_x * d_K[u, v].
void assemble_tda_ao_potential_device(int nao, const double* d_J, const double* d_Fxc,
                                      double alpha_x, const double* d_K, double* d_V_out,
                                      StreamHandle stream = nullptr);

// Assembles Full TDDFT / TDHF AO effective potentials from single non-symmetric M = C_o Z C_v^T:
//   d_Vp[u, v] = 4 * d_J[u, v] + (has_fxc ? 2 * d_Fxc[u, v] : 0) - alpha_x * (d_K[u, v] + d_K[v, u])
//   d_Vm[u, v] = -alpha_x * (d_K[u, v] - d_K[v, u])   (if d_Vm_out != nullptr)
void assemble_rpa_ao_potentials_device(int nao, const double* d_J, const double* d_Fxc,
                                       double alpha_x, const double* d_K, double* d_Vp_out,
                                       double* d_Vm_out, StreamHandle stream = nullptr);

// Elementwise diagonal scaling / addition on MO [n_ov] vectors:
//   d_y[i] = d_eps[i] * d_x[i] + beta * d_y[i]
void add_diag_eps_mo_device(int n_ov, const double* d_eps, const double* d_x, double beta,
                            double* d_y, StreamHandle stream = nullptr);

// Elementwise scaling d_y[i] = d_scale[i] * d_x[i] (used for Casida \Delta\eps^{1/2} wrapping).
void elemwise_mul_device(int n, const double* d_scale, const double* d_x, double* d_y,
                         StreamHandle stream = nullptr);
#endif

}  // namespace detail

}  // namespace gpu4dft
