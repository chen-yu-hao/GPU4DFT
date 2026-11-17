#pragma once

#include <math.h>
#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu4dft/config.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft::pw {

using Complex = std::complex<double>;

enum class PwSolverType : int {
  Davidson = 0,
  LOBPCG = 1,
  BlockDavidson = 0,
};

enum class ExxDiv : int {
  None = 0,   // Set G=0 exchange kernel to 0 (matches pyscf exxdiv=None)
  Ewald = 1,  // Gygi-Baldereschi probe-charge Madelung G=0 term (matches pyscf exxdiv='ewald')
};

// Goedecker-Teter-Hutter (GTH / HGH) non-local angular momentum channel l.
struct GthNonLocalChannel {
  int l = 0;
  double r_l = 0.0;
  int n_proj = 0;                 // 0..3 projectors
  std::array<double, 9> h{};      // row-major [n_proj x n_proj] coupling matrix h_{ij}^{(l)}
};

// Goedecker-Teter-Hutter (GTH / HGH) norm-conserving separable pseudopotential.
struct GthPseudopotential {
  std::string symbol;
  int atomic_number = 0;
  double z_ion = 0.0;             // Valence ionic charge Z_ion
  double r_loc = 0.0;             // Local pseudopotential radius R_loc (Bohr)
  std::array<double, 4> c_loc{};  // Local Gaussian polynomial coefficients c1..c4
  std::vector<GthNonLocalChannel> nl_channels;

  // G=0 pseudopotential + Hartree neutralization core constant per ion (Bohr^3 * Eh):
  //   eps_core = 2*pi * R_loc^2 * Z_ion + (2*pi)^(3/2) * R_loc^3 * (c1 + 3*c2 + 15*c3 + 105*c4)
  // Contribution to V_loc(G=0) is eps_core / Omega.
  double core_energy_constant() const noexcept;
};

// Lookup built-in GTH-LDA ('gth-pade' / 'gth-lda') or GTH-PBE ('gth-pbe') parameters
// matching PySCF's built-in GTH table for H..Ar (Z = 1..18).
GthPseudopotential lookup_gth_pseudopotential(int atomic_number,
                                              std::string_view family = "gth-pade");
GthPseudopotential lookup_gth_pseudopotential(std::string_view symbol,
                                              std::string_view family = "gth-pade");

// Element symbol <-> atomic number helpers (Z = 1..18+).
int element_atomic_number(std::string_view symbol);
std::string element_symbol(int atomic_number);

struct PwAtom {
  int atomic_number = 0;
  std::string symbol;
  std::array<double, 3> pos{};  // Cartesian coordinates in Bohr
  GthPseudopotential pseudo;    // Auto-populated by PwCell if z_ion <= 0 and pseudo_family != "none"
};

// Periodic crystal unit cell: arbitrary 3x3 lattice vectors, reciprocal lattice,
// structure factors, exact 3D Ewald ion-ion repulsion, and Madelung constant.
class PwCell {
 public:
  PwCell() = default;
  PwCell(const std::array<std::array<double, 3>, 3>& a_vecs,
         std::vector<PwAtom> atoms,
         std::string_view pseudo_family = "gth-pade");

  // Rows are lattice vectors a_1, a_2, a_3 in Bohr.
  const std::array<std::array<double, 3>, 3>& lattice_vectors() const noexcept { return a_vecs_; }

  // Rows are reciprocal lattice vectors b_1, b_2, b_3 in 1/Bohr (a_i . b_j = 2*pi * delta_{ij}).
  const std::array<std::array<double, 3>, 3>& reciprocal_vectors() const noexcept { return b_vecs_; }

  double volume() const noexcept { return volume_; }
  const std::vector<PwAtom>& atoms() const noexcept { return atoms_; }
  int natom() const noexcept { return static_cast<int>(atoms_.size()); }
  const std::string& pseudo_family() const noexcept { return pseudo_family_; }

  double total_valence_charge() const noexcept;
  int num_electrons(int charge = 0) const;

  // Exact 3D Ewald periodic ion-ion interaction energy (matches pyscf.pbc.gto.Cell.ewald() < 1e-12 Eh).
  double ewald_energy(double precision = 1e-13) const;

  // Exact Madelung constant xi = -2 * E_ewald(Z=1 in Nk supercell) for integrable 1/|G|^2 exchange divergence.
  double madelung_constant(std::array<int, 3> kmesh = {1, 1, 1}, double precision = 1e-13) const;

  // Structure factor S_I(G) = exp(-i G . R_I) on a flattened [nG * 3] array of G vectors.
  std::vector<Complex> structure_factor(int atom_index, std::span<const double> g_vecs_xyz) const;

  // Uniform real-space FFT grid coordinates [N1*N2*N3 * 3] in Bohr (row-major i1, i2, i3).
  std::vector<double> uniform_grid_coords(std::array<int, 3> mesh) const;

  // Generates Monkhorst-Pack k-points in Cartesian 1/Bohr and uniform weights summing to 1.
  // When `gamma_centered == true` and `wrap_around == false`, matches `pyscf.pbc.gto.Cell.make_kpts(kmesh)`.
  std::pair<std::vector<std::array<double, 3>>, std::vector<double>>
  monkhorst_pack_kpoints(std::array<int, 3> kmesh,
                         bool gamma_centered = true,
                         bool wrap_around = false) const;

 private:
  std::array<std::array<double, 3>, 3> a_vecs_{};
  std::array<std::array<double, 3>, 3> b_vecs_{};
  double volume_ = 0.0;
  std::vector<PwAtom> atoms_;
  std::string pseudo_family_ = "gth-pade";
};

// Smallest 2^a * 3^b * 5^c smooth integer >= n for optimal cuFFT / mixed-radix FFT throughput.
int next_fft_smooth_size(int n) noexcept;

// Active spherical plane-wave basis 0.5 * |k + G|^2 <= E_cut for a single k-point.
struct KPointBasis {
  std::array<double, 3> k_cart{};   // Cartesian k-vector (1/Bohr)
  double weight = 1.0;              // Brillouin-zone integration weight
  int npw = 0;                      // Number of active plane waves
  std::vector<int> fft_indices;     // [npw] Gather/scatter index into [N1 * N2 * N3] FFT box
  std::vector<double> kin;          // [npw] 0.5 * |k + G|^2 (Hartree)
  std::vector<double> kg_xyz;       // [npw * 3] Cartesian (k + G)
  std::vector<double> g_xyz;        // [npw * 3] Cartesian G
};

// Dual-grid Plane-Wave Basis: smooth FFT density box (E_cut^rho = 4 * E_cut)
// + per-k-point active spherical wavefunction basis (0.5 * |k + G|^2 <= E_cut).
class PwBasis {
 public:
  PwBasis(const PwCell& cell,
          double ecut_wfc,
          double ecut_rho = 0.0,
          std::span<const std::array<double, 3>> kpoints = {},
          std::span<const double> kweights = {},
          std::array<int, 3> fft_grid_override = {0, 0, 0});

  const PwCell& cell() const noexcept { return cell_; }
  double ecut_wfc() const noexcept { return ecut_wfc_; }
  double ecut_rho() const noexcept { return ecut_rho_; }
  std::array<int, 3> fft_grid() const noexcept { return fft_grid_; }
  int ngrid() const noexcept { return ngrid_; }
  int nkpts() const noexcept { return static_cast<int>(kpts_.size()); }
  const KPointBasis& kpoint(int ik = 0) const { return kpts_.at(static_cast<std::size_t>(ik)); }
  const std::vector<KPointBasis>& kpoints() const noexcept { return kpts_; }

  const std::vector<double>& g_vecs_xyz() const noexcept { return g_vecs_xyz_; }  // [ngrid * 3]
  const std::vector<double>& g2() const noexcept { return g2_; }                  // [ngrid]

  // Evaluates static ionic local pseudopotential V_loc^ion(r) on the [N1*N2*N3] real-space grid.
  std::vector<double> compute_local_pp_grid() const;

  // Wraps (q + G) vectors into minimum-image Brillouin zone counterparts (matching PySCF _Gv_wrap_around)
  // and returns Coulomb exchange kernel v_exx(q + G) [ngrid] for ExxDiv::None or ExxDiv::Ewald.
  std::vector<double> compute_exx_coulomb_kernel(const std::array<double, 3>& q_shift,
                                                 ExxDiv exxdiv,
                                                 std::array<int, 3> kmesh = {1, 1, 1}) const;

 private:
  PwCell cell_;
  double ecut_wfc_ = 0.0;
  double ecut_rho_ = 0.0;
  std::array<int, 3> fft_grid_{0, 0, 0};
  int ngrid_ = 0;
  std::vector<double> g_vecs_xyz_;  // [ngrid * 3]
  std::vector<double> g2_;          // [ngrid]
  std::vector<KPointBasis> kpts_;
};

// Separable GTH/Kleinman-Bylander Non-Local Pseudopotential Projector at a given k-point:
//   V_NL^(k) = P_k * H_proj * P_k^H,  where P_k is column-major [npw x nproj].
class NonLocalProjector {
 public:
  NonLocalProjector() = default;
  NonLocalProjector(const PwCell& cell, const KPointBasis& kp);

  int npw() const noexcept { return npw_; }
  int nproj() const noexcept { return nproj_; }
  const std::vector<Complex>& projectors_col_major() const noexcept { return P_; }  // [npw * nproj]
  const std::vector<Complex>& coupling_matrix_col_major() const noexcept { return H_; }  // [nproj * nproj]

  // Host application: H_psi += P * (H_proj * (P^H * Psi)), column-major [npw x nband].
  void apply_add_host(const Complex* psi, Complex* hpsi, int nband) const;

 private:
  int npw_ = 0;
  int nproj_ = 0;
  std::vector<Complex> P_;  // Column-major [npw x nproj]
  std::vector<Complex> H_;  // Column-major [nproj x nproj]
};

// Batched 3D complex-to-complex FFT on Host (mixed-radix 2/3/5 Cooley-Tukey + Bluestein)
// or CUDA (`cufftPlanMany` Z2Z).
//   direction = -1: FORWARD  F[m] = sum_r f[r] exp(-2*pi*i m.r/N)  (unnormalized, matches CUFFT_FORWARD)
//   direction = +1: INVERSE  f[r] = sum_m F[m] exp(+2*pi*i m.r/N)  (unnormalized, matches CUFFT_INVERSE)
void fft3d_batch(std::array<int, 3> grid,
                 int nbatch,
                 const Complex* in,
                 Complex* out,
                 int direction,
                 Backend backend = Backend::Host);

// Standalone Plane-Wave Hamiltonian eigensolver for -1/2 nabla^2 + V_loc(r) (+ optional V_NL):
// Solves lowest `nband` eigenpairs at k-point `ik` using Block-Davidson or LOBPCG on Host or CUDA.
struct PwEigenResult {
  int iterations = 0;
  int hpsi_calls = 0;
  double max_residual = 0.0;
  std::vector<double> eigenvalues;   // [nband]
  std::vector<Complex> eigenvectors; // Column-major [npw x nband]
};

PwEigenResult solve_pw_eigenproblem(const PwBasis& basis,
                                    int ik,
                                    std::span<const double> vloc_r,
                                    int nband,
                                    PwSolverType solver = PwSolverType::Davidson,
                                    Backend backend = Backend::Cuda,
                                    double residual_tol = 1e-10,
                                    int max_iter = 80,
                                    bool include_nonlocal_pp = false);

struct PwScfOptions {
  Backend backend = Backend::Cuda;
  PwSolverType eigensolver = PwSolverType::Davidson;
  std::string xc = "lda";                // e.g. "lda"/"svwn", "spz81"/"lda,pz", "pbe", "blyp", "pbe0", "b3lyp", "hf"
  ExxDiv exxdiv = ExxDiv::Ewald;         // Exact exchange G=0 singularity treatment
  std::array<int, 3> kmesh = {1, 1, 1};  // Supercell k-mesh dimensions for Madelung constant
  bool use_ace = true;                   // Lin's Adaptive Compressed Exchange (ACE) for hybrids/HF
  int charge = 0;
  int nbands = 0;                        // 0 => automatic (n_occ + buffer)
  int max_scf_iter = 80;
  int max_ace_iter = 20;
  int max_eig_iter = 50;
  double conv_tol = 1e-9;                // Total energy convergence threshold (Eh)
  double density_tol = 1e-7;             // RMS density residual threshold
  double eig_tol = 1e-10;                // Inner Davidson/LOBPCG residual tolerance
  double mixing_beta = 0.6;              // Kerker-preconditioned Pulay DIIS mixing parameter
  double kerker_q0 = 0.8;                // Kerker screening wavevector q0 (1/Bohr)
  int diis_max_vec = 8;                  // Pulay DIIS history length
  double rho_cutoff = 1e-15;             // Symbolic XC density threshold
};

struct PwScfResult {
  bool converged = false;
  int scf_iterations = 0;
  int ace_iterations = 0;
  int total_eig_iterations = 0;
  int total_hpsi_calls = 0;
  int total_poisson_ffts = 0;
  double elapsed_ms = 0.0;

  double total_energy = 0.0;
  double kinetic_energy = 0.0;
  double local_pp_energy = 0.0;
  double nonlocal_pp_energy = 0.0;
  double hartree_energy = 0.0;
  double xc_energy = 0.0;
  double exx_energy = 0.0;       // Scaled by alpha_x: alpha_x * E_X
  double ewald_energy = 0.0;
  double homo_energy = 0.0;
  double lumo_energy = 0.0;
  double final_density_rms = 0.0;

  int nbands = 0;
  std::vector<std::vector<double>> eigenvalues;  // [nkpts][nbands]
  std::vector<double> density_r;                 // [ngrid] converged real-space density
};

class PwKohnShamSolver {
 public:
  PwKohnShamSolver(const PwBasis& basis, PwScfOptions opts = {});
  ~PwKohnShamSolver();

  PwScfResult run();

 private:
  PwBasis basis_;
  PwScfOptions opts_;
};

// Check whether CUDA PW backend is compiled in and at least one GPU device is visible.
bool cuda_pw_available() noexcept;

namespace detail {

// Real solid harmonics S_{lm}(qx, qy, qz) = |q|^l Y_{lm}^{real}(q_hat) up to l = 3 (0 <= m < 2l+1).
GPU4DFT_HD inline void eval_solid_harmonics_pw(int l, double x, double y, double z, double* out) {
  constexpr double kInvSqrt4Pi = 0.28209479177387814347;
  constexpr double kSqrt3Over4Pi = 0.48860251190291992159;
  constexpr double kSqrt15Over4Pi = 1.09254843059207907054;
  constexpr double kHalfSqrt5Over4Pi = 0.31539156525252000603;
  constexpr double kPi = 3.14159265358979323846;

  if (l == 0) {
    out[0] = kInvSqrt4Pi;
    return;
  }
  if (l == 1) {
    out[0] = kSqrt3Over4Pi * x;
    out[1] = kSqrt3Over4Pi * y;
    out[2] = kSqrt3Over4Pi * z;
    return;
  }
  const double r2 = x * x + y * y + z * z;
  if (l == 2) {
    out[0] = kSqrt15Over4Pi * x * y;
    out[1] = kSqrt15Over4Pi * y * z;
    out[2] = kHalfSqrt5Over4Pi * (3.0 * z * z - r2);
    out[3] = kSqrt15Over4Pi * x * z;
    out[4] = 0.5 * kSqrt15Over4Pi * (x * x - y * y);
    return;
  }
  double c_mm = 1.0, s_mm = 0.0;
  for (int m = 0; m <= l; ++m) {
    if (m > 0) {
      const double f = static_cast<double>(2 * m - 1);
      const double nc = f * (x * c_mm - y * s_mm);
      const double ns = f * (x * s_mm + y * c_mm);
      c_mm = nc;
      s_mm = ns;
    }
    double c_prev2 = 0.0, s_prev2 = 0.0;
    double c_cur = c_mm, s_cur = s_mm;
    for (int n = m + 1; n <= l; ++n) {
      const double fn = static_cast<double>(2 * n - 1);
      double c_next, s_next;
      if (n == m + 1) {
        c_next = fn * z * c_cur;
        s_next = fn * z * s_cur;
      } else {
        const double inv_nm = 1.0 / static_cast<double>(n - m);
        const double fp = static_cast<double>(n + m - 1);
        c_next = (fn * z * c_cur - fp * r2 * c_prev2) * inv_nm;
        s_next = (fn * z * s_cur - fp * r2 * s_prev2) * inv_nm;
      }
      c_prev2 = c_cur;
      s_prev2 = s_cur;
      c_cur = c_next;
      s_cur = s_next;
    }
    if (m == 0) {
      out[l] = ::sqrt(static_cast<double>(2 * l + 1) / (4.0 * kPi)) * c_cur;
    } else {
      double fact_ratio = 1.0;
      for (int k = l - m + 1; k <= l + m; ++k) fact_ratio *= static_cast<double>(k);
      const double norm_m = ::sqrt(static_cast<double>(2 * l + 1) / (2.0 * kPi * fact_ratio));
      out[l - m] = norm_m * s_cur;
      out[l + m] = norm_m * c_cur;
    }
  }
}

// Exact GTH/HGH Fourier radial polynomial q_{l,i}(x) where x = |k + G| * r_l.
GPU4DFT_HD inline double gth_qli(double x, int l, int i) {
  const double x2 = x * x;
  const double x4 = x2 * x2;
  if (l == 0) {
    if (i == 0) return 4.0 * ::sqrt(2.0);
    if (i == 1) return 8.0 * ::sqrt(2.0 / 15.0) * (3.0 - x2);
    if (i == 2) return (16.0 / 3.0) * ::sqrt(2.0 / 105.0) * (15.0 - 10.0 * x2 + x4);
  } else if (l == 1) {
    if (i == 0) return 8.0 * ::sqrt(1.0 / 3.0);
    if (i == 1) return 16.0 * ::sqrt(1.0 / 105.0) * (5.0 - x2);
    if (i == 2) return (32.0 / 3.0) * ::sqrt(1.0 / 1155.0) * (35.0 - 14.0 * x2 + x4);
  } else if (l == 2) {
    if (i == 0) return 8.0 * ::sqrt(2.0 / 15.0);
    if (i == 1) return (16.0 / 3.0) * ::sqrt(2.0 / 105.0) * (7.0 - x2);
    if (i == 2) return (32.0 / 3.0) * ::sqrt(2.0 / 15015.0) * (63.0 - 18.0 * x2 + x4);
  } else if (l == 3) {
    if (i == 0) return 16.0 * ::sqrt(1.0 / 105.0);
    if (i == 1) return (32.0 / 3.0) * ::sqrt(1.0 / 1155.0) * (9.0 - x2);
    if (i == 2) return (64.0 / 45.0) * ::sqrt(1.0 / 1001.0) * (99.0 - 22.0 * x2 + x4);
  }
  return 0.0;
}

// Teter-Payne-Allan (PRB 40, 12255, 1989) smooth rational kinetic preconditioner.
GPU4DFT_HD inline double teter_preconditioner(double kin_g, double ekin_band) {
  const double ek = (ekin_band > 0.5) ? ekin_band : 0.5;
  const double denom_scale = 1.5 * ek;
  const double x = kin_g / denom_scale;
  const double x2 = x * x;
  const double x3 = x2 * x;
  const double x4 = x2 * x2;
  const double num = 27.0 + 18.0 * x + 12.0 * x2 + 8.0 * x3;
  const double den = num + 16.0 * x4;
  return (num / den) / denom_scale;
}

void pw_zheev_host(int n, const Complex* a_col_major, double* w_out, Complex* u_col_major);
void pw_cholesky_trsm_ace_host(int m, int n, const Complex* b_in, const Complex* w_in, Complex* xi_out);

#if defined(GPU4DFT_HAVE_CUDA)
int pw_cuda_device_count() noexcept;

void pw_cuda_fft3d_batch(std::array<int, 3> grid,
                         int nbatch,
                         const Complex* h_in,
                         Complex* h_out,
                         int direction);

PwEigenResult pw_cuda_solve_eigenproblem(const PwBasis& basis,
                                         int ik,
                                         std::span<const double> vloc_r,
                                         int nband,
                                         PwSolverType solver,
                                         double residual_tol,
                                         int max_iter,
                                         bool include_nonlocal_pp);

PwScfResult pw_cuda_run_scf(const PwBasis& basis, const PwScfOptions& opts);
#endif

}  // namespace detail

}  // namespace gpu4dft::pw
