#pragma once

#include <math.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/config.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

struct XcAtom {
  int atomic_number = 0;
  std::array<double, 3> center{};  // Bohr
};

enum class PartitionScheme {
  Stratmann,  // Stratmann-Scuseria-Frisch (SSF, a = 0.64) smooth localized partitioning
  Becke,      // Becke 3-iteration polynomial partitioning
};

struct GridOptions {
  int level = 3;   // Grid quality preset in [0, 5] when n_rad / n_ang == 0
  int n_rad = 0;   // 0 = automatic per element row from level; >0 = explicit radial shells
  int n_ang = 0;   // 0 = automatic from level (with pruning if prune=true); >0 = explicit Lebedev size (14,26,50,110,194,302)
  bool prune = true;
  PartitionScheme partition = PartitionScheme::Stratmann;
  bool becke_radii_adjust = true;
  double weight_cutoff = 1e-14;
  int max_batch_points = 2048;
  double ao_cutoff = 1e-12;
  Backend backend = Backend::Cuda;  // GPU Becke/SSF weights when Cuda is available, else Host OpenMP
};

struct SpatialBatch {
  std::size_t point_offset = 0;
  std::size_t npoints = 0;
  std::array<double, 3> box_min{};
  std::array<double, 3> box_max{};
  std::vector<int> active_shells;
  std::vector<int> active_aos;
};

class MolecularGrid {
 public:
  MolecularGrid() = default;

  // Generates Treutler-Ahlrichs (M4) radial + Lebedev angular grids, computes
  // Becke / Stratmann (SSF) atomic weights on GPU (or Host OpenMP), reorders
  // points into 3D octree spatial batches, and screens active shells/AOs per batch.
  MolecularGrid(std::span<const XcAtom> atoms, const Basis& basis,
                const GridOptions& opts = {});

  // Builds a spatially batched + active-AO-screened MolecularGrid from external
  // quadrature points (e.g. PySCF grids.coords [N*3] in Bohr and grids.weights [N]).
  static MolecularGrid from_points(std::span<const double> coords_xyz,
                                   std::span<const double> weights,
                                   const Basis& basis,
                                   int max_batch_points = 2048,
                                   double ao_cutoff = 1e-12);

  // Returns unit-sphere Lebedev quadrature (coords_xyz [n_ang * 3], weights [n_ang]
  // summing to 4*pi) for supported sizes {14, 26, 50, 110, 194, 302}.
  static std::pair<std::vector<double>, std::vector<double>> lebedev_grid(int n_ang);

  // Computes radial extent R_shell (Bohr) per shell in `basis` for threshold `ao_cutoff`.
  static std::vector<double> compute_shell_radii(const Basis& basis, double ao_cutoff = 1e-12);

  std::size_t npoints() const { return weights_.size(); }
  const std::vector<double>& coords() const { return coords_; }    // [npoints * 3] row-major (x,y,z)
  const std::vector<double>& weights() const { return weights_; }  // [npoints]
  const std::vector<SpatialBatch>& batches() const { return batches_; }
  std::size_t nbatches() const { return batches_.size(); }
  int nao() const { return nao_; }

 private:
  void finalize_spatial_batches(std::vector<double> raw_coords,
                                std::vector<double> raw_weights,
                                const Basis& basis,
                                int max_batch_points,
                                double ao_cutoff);

  std::vector<double> coords_;
  std::vector<double> weights_;
  std::vector<SpatialBatch> batches_;
  int nao_ = 0;
};

// Returns exact Hartree-Fock exchange fraction alpha_x (e.g. 0.20 for B3LYP,
// 0.25 for PBE0, 0.0 for pure LDA/GGA/TPSS, 1.0 for HF).
double xc_exact_exchange_fraction(std::string_view xc_name);

// Point-wise evaluation result for generated symbolic XC functionals (deriv = 1 or 2).
struct XcPointwiseResult {
  std::vector<double> zk;
  std::vector<double> vrho;
  std::vector<double> vsigma;
  std::vector<double> vtau;
  std::vector<double> v2rho2;
  std::vector<double> v2rhosigma;
  std::vector<double> v2sigma2;
  std::vector<double> v2rhotau;
  std::vector<double> v2sigmatau;
  std::vector<double> v2tau2;
};

// Evaluates generated symbolic XC kernel point-wise on Host or CUDA GPU.
XcPointwiseResult eval_xc_points(std::string_view xc_name,
                                 std::span<const double> rho,
                                 std::span<const double> sigma = {},
                                 std::span<const double> tau = {},
                                 int deriv = 1,
                                 Backend backend = Backend::Host,
                                 double rho_cutoff = 1e-15);

// Evaluates spherical AOs (and derivatives up to `deriv` in {0, 1, 2}) at `coords_xyz` [N*3].
// Returns flattened [ncomp * npts * nao] matching PySCF's eval_ao component ordering:
//   comp 0: phi;  1..3: x, y, z;  4..9: xx, xy, xz, yy, yz, zz.
std::vector<double> eval_ao_grid(const Basis& basis,
                                 std::span<const double> coords_xyz,
                                 int deriv = 0,
                                 Backend backend = Backend::Host);

struct XcOptions {
  Backend backend = Backend::Cuda;
  Precision precision = Precision::FP64;  // FP64: cublasDgemm; FP32: Tensor Core cublasSgemm half-transforms + FP64 accum
  double rho_cutoff = 1e-15;
  bool cache_ao_on_device = true;         // Cache compacted active-AO Phi across batches on GPU when within VRAM cap
  std::size_t max_ao_cache_mb = 2048;     // Maximum GPU memory budget (MiB) for persistent active-AO Phi cache
};

struct XcResult {
  double exc = 0.0;
  double nelec = 0.0;
  std::vector<double> vxc;  // [nao * nao] row-major symmetric matrix
};

class XcEngine {
 public:
  XcEngine(const Basis& basis, const MolecularGrid& grid,
           std::string_view xc_name, const XcOptions& opts = {});
  ~XcEngine();

  XcEngine(XcEngine&&) noexcept;
  XcEngine& operator=(XcEngine&&) noexcept;
  XcEngine(const XcEngine&) = delete;
  XcEngine& operator=(const XcEngine&) = delete;

  int nao() const { return basis_.nao(); }
  const std::string& xc_name() const { return xc_name_; }
  double exact_exchange_fraction() const { return exx_fraction_; }
  const MolecularGrid& grid() const { return grid_; }
  Backend active_backend() const { return active_backend_; }

  // Host API: evaluates E_xc, integrated electron count, and [nao x nao] V_xc.
  XcResult eval_vxc(std::span<const double> dm) const;
  double eval_vxc(const double* dm, double* vxc_out, double* nelec_out = nullptr) const;

  // GPU Device-Pointer API: takes device pointers `d_dm` [nao * nao] and `d_vxc` [nao * nao]
  // (writes symmetric V_xc on GPU without host matrix transfers) and returns E_xc.
  double eval_vxc_device(const double* d_dm, double* d_vxc, void* cuda_stream = nullptr) const;

  // Second-Derivative XC Linear Response (TDDFT / CPHF Hessian contraction f_xc[rho_0]):
  // Precomputes and caches ground-state (rho_0, grad_rho_0, tau_0) and 1st+2nd XC functional
  // derivatives across all grid points on Host or GPU.
  void set_reference_density(const double* h_dm);
  void set_reference_density(std::span<const double> h_dm);
  void set_reference_density_device(const double* d_dm, void* cuda_stream = nullptr);
  bool has_reference_density() const noexcept;

  // Contracts a symmetric (or general real, automatically symmetrized via 0.5*(dD + dD^T))
  // perturbation/transition density delta_D [nao x nao] with the cached ground-state f_xc[rho_0]
  // kernel via batched GEMMs, returning [nao x nao] F_xc[delta_D] matching pyscf.dft.numint.nr_rks_fxc.
  std::vector<double> eval_fxc(const double* h_delta_dm) const;
  std::vector<double> eval_fxc(std::span<const double> h_delta_dm) const;
  void eval_fxc(const double* h_delta_dm, double* h_fxc_out) const;
  void eval_fxc_device(const double* d_delta_dm, double* d_fxc_out,
                       void* cuda_stream = nullptr) const;

  // Analytical XC nuclear gradient AO-derivative term [natom * 3] matching PySCF rks.Gradients().
  std::vector<double> nuclear_grad(std::span<const XcAtom> atoms,
                                   std::span<const double> dm) const;

 private:
  Basis basis_;
  MolecularGrid grid_;
  std::string xc_name_;
  XcOptions opts_;
  Backend active_backend_ = Backend::Host;
  int xc_func_id_ = 0;
  int xc_family_ = 0;  // 0=LDA, 1=GGA, 2=MGGA
  double exx_fraction_ = 0.0;

  struct HostFxcState;
  std::unique_ptr<HostFxcState> host_fxc_;

  struct CudaState;
  std::unique_ptr<CudaState> cuda_;
};

// Convenience free function for analytical XC nuclear gradient [natom * 3] (Hartree / Bohr).
std::vector<double> xc_nuclear_grad(const Basis& basis,
                                    const MolecularGrid& grid,
                                    std::span<const XcAtom> atoms,
                                    std::string_view xc_name,
                                    const double* dm,
                                    Backend backend = Backend::Cuda);

namespace detail {

struct FlatBasisView {
  int nshell = 0;
  int nao = 0;
  const int* shell_l = nullptr;
  const int* shell_ao_offset = nullptr;
  const int* shell_prim_offset = nullptr;
  const int* shell_prim_count = nullptr;
  const double* shell_center = nullptr;  // [nshell * 3]
  const double* prim_exp = nullptr;
  const double* prim_coeff = nullptr;
};

struct FlatBasisData {
  int nshell = 0;
  int nao = 0;
  std::vector<int> shell_l;
  std::vector<int> shell_ao_offset;
  std::vector<int> shell_prim_offset;
  std::vector<int> shell_prim_count;
  std::vector<double> shell_center;
  std::vector<double> prim_exp;
  std::vector<double> prim_coeff;

  explicit FlatBasisData(const Basis& basis)
      : nshell(static_cast<int>(basis.nshell())), nao(basis.nao()) {
    shell_l.resize(basis.nshell());
    shell_ao_offset.resize(basis.nshell());
    shell_prim_offset.resize(basis.nshell());
    shell_prim_count.resize(basis.nshell());
    shell_center.resize(basis.nshell() * 3);
    for (std::size_t is = 0; is < basis.nshell(); ++is) {
      const Shell& sh = basis.shell(is);
      shell_l[is] = sh.l;
      shell_ao_offset[is] = basis.ao_offset(is);
      shell_prim_offset[is] = static_cast<int>(prim_exp.size());
      shell_prim_count[is] = static_cast<int>(sh.exponents.size());
      shell_center[3 * is + 0] = sh.center[0];
      shell_center[3 * is + 1] = sh.center[1];
      shell_center[3 * is + 2] = sh.center[2];
      for (std::size_t k = 0; k < sh.exponents.size(); ++k) {
        prim_exp.push_back(sh.exponents[k]);
        prim_coeff.push_back(sh.coefficients[k]);
      }
    }
  }

  FlatBasisView view() const {
    return FlatBasisView{
        nshell,
        nao,
        shell_l.data(),
        shell_ao_offset.data(),
        shell_prim_offset.data(),
        shell_prim_count.data(),
        shell_center.data(),
        prim_exp.data(),
        prim_coeff.data(),
    };
  }
};

struct SuperBatch {
  std::size_t point_offset = 0;
  std::size_t npoints = 0;
  std::vector<int> active_shells;
  std::vector<int> active_aos;
};

// Coalesces consecutive Morton-ordered SpatialBatches into high-occupancy GPU super-batches
// up to `target_max_points` whenever points are contiguous and active-AO union remains tight.
inline std::vector<SuperBatch> build_super_batches(
    const MolecularGrid& grid,
    const FlatBasisData& flat_b,
    std::size_t target_max_points = 2048,
    double max_work_expansion = 1.40) {
  std::vector<SuperBatch> out;
  const auto& src = grid.batches();
  if (src.empty()) return out;

  auto count_aos_for_shells = [&](const std::vector<int>& shells) -> std::size_t {
    std::size_t n = 0;
    for (int is : shells) {
      n += static_cast<std::size_t>(2 * flat_b.shell_l[static_cast<std::size_t>(is)] + 1);
    }
    return n;
  };

  auto populate_aos = [&](SuperBatch& sb) {
    sb.active_aos.clear();
    for (int is : sb.active_shells) {
      const int ao0 = flat_b.shell_ao_offset[static_cast<std::size_t>(is)];
      const int nm = 2 * flat_b.shell_l[static_cast<std::size_t>(is)] + 1;
      for (int m = 0; m < nm; ++m) sb.active_aos.push_back(ao0 + m);
    }
  };

  SuperBatch cur;
  double cur_leaf_work = 0.0;
  bool has_cur = false;
  std::vector<int> merged_shells;

  for (const SpatialBatch& b : src) {
    if (b.npoints == 0 || b.active_shells.empty()) continue;
    const double leaf_work =
        static_cast<double>(b.active_aos.size()) * static_cast<double>(b.npoints);

    if (!has_cur) {
      cur.point_offset = b.point_offset;
      cur.npoints = b.npoints;
      cur.active_shells = b.active_shells;
      cur_leaf_work = leaf_work;
      has_cur = true;
      continue;
    }

    const bool contiguous = (b.point_offset == cur.point_offset + cur.npoints);
    const std::size_t cand_ng = cur.npoints + b.npoints;
    bool do_merge = false;
    if (contiguous && cand_ng <= target_max_points) {
      merged_shells.clear();
      merged_shells.reserve(cur.active_shells.size() + b.active_shells.size());
      std::set_union(cur.active_shells.begin(), cur.active_shells.end(),
                     b.active_shells.begin(), b.active_shells.end(),
                     std::back_inserter(merged_shells));
      const std::size_t cand_na = count_aos_for_shells(merged_shells);
      const double cand_work = static_cast<double>(cand_na) * static_cast<double>(cand_ng);
      const double base_work = cur_leaf_work + leaf_work;
      if (cand_work <= max_work_expansion * base_work ||
          (cur.npoints < 256 && cand_ng <= 768 && cand_work <= 1.80 * base_work)) {
        do_merge = true;
      }
    }

    if (do_merge) {
      cur.npoints = cand_ng;
      cur.active_shells.swap(merged_shells);
      cur_leaf_work += leaf_work;
    } else {
      populate_aos(cur);
      out.push_back(std::move(cur));
      cur = SuperBatch{};
      cur.point_offset = b.point_offset;
      cur.npoints = b.npoints;
      cur.active_shells = b.active_shells;
      cur_leaf_work = leaf_work;
    }
  }

  if (has_cur) {
    populate_aos(cur);
    out.push_back(std::move(cur));
  }
  return out;
}

// Second-order forward-mode Jet over (x, y, z) for exact solid-harmonic derivatives (0 <= l <= 7).
struct Jet3D2 {
  double v = 0.0;
  double dx = 0.0, dy = 0.0, dz = 0.0;
  double dxx = 0.0, dxy = 0.0, dxz = 0.0, dyy = 0.0, dyz = 0.0, dzz = 0.0;

  GPU4DFT_HD static constexpr Jet3D2 constant(double c) { return Jet3D2{c}; }
  GPU4DFT_HD static constexpr Jet3D2 var_x(double x) {
    Jet3D2 j{};
    j.v = x;
    j.dx = 1.0;
    return j;
  }
  GPU4DFT_HD static constexpr Jet3D2 var_y(double y) {
    Jet3D2 j{};
    j.v = y;
    j.dy = 1.0;
    return j;
  }
  GPU4DFT_HD static constexpr Jet3D2 var_z(double z) {
    Jet3D2 j{};
    j.v = z;
    j.dz = 1.0;
    return j;
  }

  GPU4DFT_HD constexpr Jet3D2 operator+(const Jet3D2& o) const {
    return Jet3D2{v + o.v,     dx + o.dx,   dy + o.dy,   dz + o.dz,   dxx + o.dxx,
                  dxy + o.dxy, dxz + o.dxz, dyy + o.dyy, dyz + o.dyz, dzz + o.dzz};
  }
  GPU4DFT_HD constexpr Jet3D2 operator-(const Jet3D2& o) const {
    return Jet3D2{v - o.v,     dx - o.dx,   dy - o.dy,   dz - o.dz,   dxx - o.dxx,
                  dxy - o.dxy, dxz - o.dxz, dyy - o.dyy, dyz - o.dyz, dzz - o.dzz};
  }
  GPU4DFT_HD constexpr Jet3D2 scale(double s) const {
    return Jet3D2{s * v,   s * dx,  s * dy,  s * dz,  s * dxx,
                  s * dxy, s * dxz, s * dyy, s * dyz, s * dzz};
  }
  GPU4DFT_HD constexpr Jet3D2 operator*(const Jet3D2& o) const {
    return Jet3D2{
        v * o.v,
        dx * o.v + v * o.dx,
        dy * o.v + v * o.dy,
        dz * o.v + v * o.dz,
        dxx * o.v + 2.0 * dx * o.dx + v * o.dxx,
        dxy * o.v + dx * o.dy + dy * o.dx + v * o.dxy,
        dxz * o.v + dx * o.dz + dz * o.dx + v * o.dxz,
        dyy * o.v + 2.0 * dy * o.dy + v * o.dyy,
        dyz * o.v + dy * o.dz + dz * o.dy + v * o.dyz,
        dzz * o.v + 2.0 * dz * o.dz + v * o.dzz,
    };
  }
};

// Evaluates real solid harmonics P_{l, idx}(x, y, z) (0 <= idx < 2l+1) in PySCF spherical order:
//   l = 0: s
//   l = 1: px, py, pz
//   l >= 2: m = -l, -l+1, ..., +l
GPU4DFT_HD inline void eval_solid_harmonics_jet(int l, double x, double y, double z,
                                                Jet3D2* out_p) {
  constexpr double kInvSqrt4Pi = 0.28209479177387814347;   // 1 / sqrt(4 * pi)
  constexpr double kSqrt3Over4Pi = 0.48860251190291992159; // sqrt(3 / (4 * pi))
  constexpr double kPi = 3.14159265358979323846;

  if (l == 0) {
    out_p[0] = Jet3D2::constant(kInvSqrt4Pi);
    return;
  }
  const Jet3D2 X = Jet3D2::var_x(x);
  const Jet3D2 Y = Jet3D2::var_y(y);
  const Jet3D2 Z = Jet3D2::var_z(z);
  if (l == 1) {
    out_p[0] = X.scale(kSqrt3Over4Pi);
    out_p[1] = Y.scale(kSqrt3Over4Pi);
    out_p[2] = Z.scale(kSqrt3Over4Pi);
    return;
  }

  const Jet3D2 R2 = X * X + Y * Y + Z * Z;
  // Diagonal recurrence for C_{m,m}, S_{m,m} up to m = l
  Jet3D2 C_mm = Jet3D2::constant(1.0);
  Jet3D2 S_mm = Jet3D2::constant(0.0);

  for (int m = 0; m <= l; ++m) {
    if (m > 0) {
      const double f = static_cast<double>(2 * m - 1);
      const Jet3D2 next_c = (X * C_mm - Y * S_mm).scale(f);
      const Jet3D2 next_s = (X * S_mm + Y * C_mm).scale(f);
      C_mm = next_c;
      S_mm = next_s;
    }

    // Vertical recurrence from n = m up to n = l
    Jet3D2 C_prev2{}, S_prev2{};
    Jet3D2 C_cur = C_mm;
    Jet3D2 S_cur = S_mm;
    for (int n = m + 1; n <= l; ++n) {
      const double fn = static_cast<double>(2 * n - 1);
      Jet3D2 C_next, S_next;
      if (n == m + 1) {
        C_next = (Z * C_cur).scale(fn);
        S_next = (Z * S_cur).scale(fn);
      } else {
        const double inv_nm = 1.0 / static_cast<double>(n - m);
        const double f_prev = static_cast<double>(n + m - 1);
        C_next = ((Z * C_cur).scale(fn) - (R2 * C_prev2).scale(f_prev)).scale(inv_nm);
        S_next = ((Z * S_cur).scale(fn) - (R2 * S_prev2).scale(f_prev)).scale(inv_nm);
      }
      C_prev2 = C_cur;
      S_prev2 = S_cur;
      C_cur = C_next;
      S_cur = S_next;
    }

    if (m == 0) {
      const double norm0 = ::sqrt(static_cast<double>(2 * l + 1) / (4.0 * kPi));
      out_p[l] = C_cur.scale(norm0);
    } else {
      double fact_ratio = 1.0;
      for (int k = l - m + 1; k <= l + m; ++k) {
        fact_ratio *= static_cast<double>(k);
      }
      const double norm_m = ::sqrt(static_cast<double>(2 * l + 1) / (2.0 * kPi * fact_ratio));
      out_p[l - m] = S_cur.scale(norm_m);
      out_p[l + m] = C_cur.scale(norm_m);
    }
  }
}

// Fast deriv=0 solid harmonic values only (used in LDA and sn-K grid AO evaluation).
GPU4DFT_HD inline void eval_solid_harmonics_d0(int l, double x, double y, double z,
                                               double* out_p) {
  constexpr double kInvSqrt4Pi = 0.28209479177387814347;
  constexpr double kSqrt3Over4Pi = 0.48860251190291992159;
  constexpr double kSqrt15Over4Pi = 1.09254843059207907054;
  constexpr double kHalfSqrt5Over4Pi = 0.31539156525252000603;
  constexpr double kPi = 3.14159265358979323846;

  if (l == 0) {
    out_p[0] = kInvSqrt4Pi;
    return;
  }
  if (l == 1) {
    out_p[0] = kSqrt3Over4Pi * x;
    out_p[1] = kSqrt3Over4Pi * y;
    out_p[2] = kSqrt3Over4Pi * z;
    return;
  }
  const double r2 = x * x + y * y + z * z;
  if (l == 2) {
    out_p[0] = kSqrt15Over4Pi * x * y;
    out_p[1] = kSqrt15Over4Pi * y * z;
    out_p[2] = kHalfSqrt5Over4Pi * (3.0 * z * z - r2);
    out_p[3] = kSqrt15Over4Pi * x * z;
    out_p[4] = 0.5 * kSqrt15Over4Pi * (x * x - y * y);
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
      out_p[l] = ::sqrt(static_cast<double>(2 * l + 1) / (4.0 * kPi)) * c_cur;
    } else {
      double fact_ratio = 1.0;
      for (int k = l - m + 1; k <= l + m; ++k) fact_ratio *= static_cast<double>(k);
      const double norm_m = ::sqrt(static_cast<double>(2 * l + 1) / (2.0 * kPi * fact_ratio));
      out_p[l - m] = norm_m * s_cur;
      out_p[l + m] = norm_m * c_cur;
    }
  }
}

// Evaluates one contracted spherical shell `ishell` at point (gx, gy, gz) up to `deriv` in {0,1,2}.
// Writes `2*l + 1` AOs into `out_comp[comp][m]` where stride between components is `comp_stride`
// and stride between consecutive `m` within this shell is `ao_stride`.
GPU4DFT_HD inline void eval_shell_at_point(const FlatBasisView& b,
                                           int ishell,
                                           double gx, double gy, double gz,
                                           int deriv,
                                           double* out_base,
                                           std::size_t comp_stride,
                                           std::size_t ao_stride) {
  const int l = b.shell_l[ishell];
  const int p0 = b.shell_prim_offset[ishell];
  const int np = b.shell_prim_count[ishell];
  const double x = gx - b.shell_center[3 * ishell + 0];
  const double y = gy - b.shell_center[3 * ishell + 1];
  const double z = gz - b.shell_center[3 * ishell + 2];
  const double r2 = x * x + y * y + z * z;
  const int nm = 2 * l + 1;

  double g0 = 0.0, g1 = 0.0, g2 = 0.0;
  for (int k = 0; k < np; ++k) {
    const double a = b.prim_exp[p0 + k];
    const double ar2 = a * r2;
    if (ar2 > 80.0) continue;
    const double ce = b.prim_coeff[p0 + k] * ::exp(-ar2);
    g0 += ce;
    if (deriv >= 1) {
      const double m2a = -2.0 * a;
      g1 += m2a * ce;
      if (deriv >= 2) {
        g2 += m2a * m2a * ce;
      }
    }
  }

  if (deriv == 0) {
    double p_vals[15];
    eval_solid_harmonics_d0(l, x, y, z, p_vals);
    for (int m = 0; m < nm; ++m) {
      out_base[m * ao_stride] = g0 * p_vals[m];
    }
    return;
  }

  Jet3D2 p_jets[15];
  eval_solid_harmonics_jet(l, x, y, z, p_jets);

  const double g1x = g1 * x;
  const double g1y = g1 * y;
  const double g1z = g1 * z;

  for (int m = 0; m < nm; ++m) {
    const Jet3D2& P = p_jets[m];
    const std::size_t off = static_cast<std::size_t>(m) * ao_stride;
    out_base[0 * comp_stride + off] = g0 * P.v;
    out_base[1 * comp_stride + off] = g1x * P.v + g0 * P.dx;
    out_base[2 * comp_stride + off] = g1y * P.v + g0 * P.dy;
    out_base[3 * comp_stride + off] = g1z * P.v + g0 * P.dz;

    if (deriv >= 2) {
      out_base[4 * comp_stride + off] = (g2 * x * x + g1) * P.v + 2.0 * g1x * P.dx + g0 * P.dxx;
      out_base[5 * comp_stride + off] = (g2 * x * y) * P.v + g1x * P.dy + g1y * P.dx + g0 * P.dxy;
      out_base[6 * comp_stride + off] = (g2 * x * z) * P.v + g1x * P.dz + g1z * P.dx + g0 * P.dxz;
      out_base[7 * comp_stride + off] = (g2 * y * y + g1) * P.v + 2.0 * g1y * P.dy + g0 * P.dyy;
      out_base[8 * comp_stride + off] = (g2 * y * z) * P.v + g1y * P.dz + g1z * P.dy + g0 * P.dyz;
      out_base[9 * comp_stride + off] = (g2 * z * z + g1) * P.v + 2.0 * g1z * P.dz + g0 * P.dzz;
    }
  }
}

}  // namespace detail

}  // namespace gpu4dft
