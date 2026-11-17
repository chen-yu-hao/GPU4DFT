#include "gpu4dft/pw.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "gpu4dft/generated/xc_kernels.hpp"

namespace gpu4dft::pw {

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kFourPi = 4.0 * kPi;

std::string to_lower_trimmed(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char ch : s) {
    if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return out;
}

GthNonLocalChannel make_nl_1(int l, double r_l, double h00) {
  GthNonLocalChannel ch{};
  ch.l = l;
  ch.r_l = r_l;
  ch.n_proj = 1;
  ch.h[0] = h00;
  return ch;
}

GthNonLocalChannel make_nl_2(int l, double r_l, double h00, double h01, double h11) {
  GthNonLocalChannel ch{};
  ch.l = l;
  ch.r_l = r_l;
  ch.n_proj = 2;
  ch.h[0] = h00;
  ch.h[1] = h01;
  ch.h[2] = h01;
  ch.h[3] = h11;
  return ch;
}

constexpr const char* kElementSymbols[19] = {
    "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",
    "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
};

}  // namespace

int element_atomic_number(std::string_view symbol) {
  const std::string key = to_lower_trimmed(symbol);
  for (int z = 1; z <= 18; ++z) {
    if (key == to_lower_trimmed(kElementSymbols[z])) return z;
  }
  throw std::invalid_argument("Unsupported element symbol in PW GTH table: " + std::string(symbol));
}

std::string element_symbol(int atomic_number) {
  if (atomic_number >= 1 && atomic_number <= 18) {
    return kElementSymbols[atomic_number];
  }
  throw std::invalid_argument("Unsupported atomic number in PW GTH table: " +
                              std::to_string(atomic_number));
}

double GthPseudopotential::core_energy_constant() const noexcept {
  const double two_pi_32 = std::pow(kTwoPi, 1.5);
  const double r2 = r_loc * r_loc;
  const double r3 = r2 * r_loc;
  return kTwoPi * r2 * z_ion +
         two_pi_32 * r3 * (c_loc[0] + 3.0 * c_loc[1] + 15.0 * c_loc[2] + 105.0 * c_loc[3]);
}

GthPseudopotential lookup_gth_pseudopotential(int atomic_number, std::string_view family) {
  const std::string fam = to_lower_trimmed(family);
  const bool is_pbe = (fam == "gth-pbe" || fam == "gth_pbe" || fam == "pbe");
  const bool is_pade =
      (fam == "gth-pade" || fam == "gth_pade" || fam == "gth-lda" || fam == "gth_lda" ||
       fam == "pade" || fam == "lda");
  if (!is_pbe && !is_pade) {
    throw std::invalid_argument("Unknown GTH pseudopotential family: " + std::string(family));
  }
  if (atomic_number < 1 || atomic_number > 18) {
    throw std::invalid_argument("Built-in GTH pseudopotentials cover Z = 1..18; got Z = " +
                                std::to_string(atomic_number));
  }

  GthPseudopotential pp{};
  pp.atomic_number = atomic_number;
  pp.symbol = kElementSymbols[atomic_number];

  if (is_pade) {
    switch (atomic_number) {
      case 1:  // H
        pp.z_ion = 1.0; pp.r_loc = 0.2; pp.c_loc = {-4.1802368, 0.72507482, 0.0, 0.0}; break;
      case 2:  // He
        pp.z_ion = 2.0; pp.r_loc = 0.2; pp.c_loc = {-9.1120234, 1.69836797, 0.0, 0.0}; break;
      case 3:  // Li
        pp.z_ion = 3.0; pp.r_loc = 0.4;
        pp.c_loc = {-14.03486849, 9.55347627, -1.76648817, 0.08436998}; break;
      case 4:  // Be
        pp.z_ion = 4.0; pp.r_loc = 0.325;
        pp.c_loc = {-24.01504092, 17.20401444, -3.32639018, 0.16541943}; break;
      case 5:  // B
        pp.z_ion = 3.0; pp.r_loc = 0.43392956; pp.c_loc = {-5.57864173, 0.80425145, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.37384326, 6.23392822)}; break;
      case 6:  // C
        pp.z_ion = 4.0; pp.r_loc = 0.34883045; pp.c_loc = {-8.5137711, 1.22843203, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.30455321, 9.52284179)}; break;
      case 7:  // N
        pp.z_ion = 5.0; pp.r_loc = 0.28917923; pp.c_loc = {-12.23481988, 1.76640728, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.25660487, 13.55224272)}; break;
      case 8:  // O
        pp.z_ion = 6.0; pp.r_loc = 0.24762086; pp.c_loc = {-16.58031797, 2.39570092, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.22178614, 18.26691718)}; break;
      case 9:  // F
        pp.z_ion = 7.0; pp.r_loc = 0.21852465; pp.c_loc = {-21.30736112, 3.07286942, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.19556721, 23.58494211)}; break;
      case 10:  // Ne
        pp.z_ion = 8.0; pp.r_loc = 0.19; pp.c_loc = {-27.69285182, 4.00590585, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.17948804, 28.50609828, 0.416828, -1.07624528),
                          make_nl_1(1, 0.21491271, -8.991e-05)}; break;
      case 11:  // Na
        pp.z_ion = 9.0; pp.r_loc = 0.2463178; pp.c_loc = {-7.54559253, 1.12599671, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.14125125, 36.55698653),
                          make_nl_1(1, 0.1396684, -10.39208332)}; break;
      case 12:  // Mg
        pp.z_ion = 10.0; pp.r_loc = 0.21094954; pp.c_loc = {-19.41900751, 2.87133099, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.14154696, 40.31662629),
                          make_nl_1(1, 0.10546902, -10.89111329)}; break;
      case 13:  // Al
        pp.z_ion = 3.0; pp.r_loc = 0.45; pp.c_loc = {-8.49135116, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.46010427, 5.08833953, -1.03784325, 2.67969975),
                          make_nl_1(1, 0.53674439, 2.19343827)}; break;
      case 14:  // Si
        pp.z_ion = 4.0; pp.r_loc = 0.44; pp.c_loc = {-7.33610297, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.42273813, 5.90692831, -1.26189397, 3.25819622),
                          make_nl_1(1, 0.48427842, 2.72701346)}; break;
      case 15:  // P
        pp.z_ion = 5.0; pp.r_loc = 0.43; pp.c_loc = {-6.65421981, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.38980284, 6.84213556, -1.4936909, 3.85669332),
                          make_nl_1(1, 0.44079585, 3.28260592)}; break;
      case 16:  // S
        pp.z_ion = 6.0; pp.r_loc = 0.42; pp.c_loc = {-6.55449184, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.36175665, 7.9053025, -1.7318813, 4.4716983),
                          make_nl_1(1, 0.40528502, 3.866579)}; break;
      case 17:  // Cl
        pp.z_ion = 7.0; pp.r_loc = 0.41; pp.c_loc = {-6.86475431, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.33820832, 9.06223968, -1.96193036, 5.0656824),
                          make_nl_1(1, 0.37613709, 4.4658764)}; break;
      case 18:  // Ar
        pp.z_ion = 8.0; pp.r_loc = 0.4; pp.c_loc = {-7.1, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.31738081, 10.24948699, -2.16984522, 5.60251627),
                          make_nl_1(1, 0.35161921, 4.97880101)}; break;
    }
  } else {
    switch (atomic_number) {
      case 1:  // H
        pp.z_ion = 1.0; pp.r_loc = 0.2; pp.c_loc = {-4.17890044, 0.72446331, 0.0, 0.0}; break;
      case 2:  // He
        pp.z_ion = 2.0; pp.r_loc = 0.2; pp.c_loc = {-9.12214383, 1.7027077, 0.0, 0.0}; break;
      case 3:  // Li
        pp.z_ion = 3.0; pp.r_loc = 0.4;
        pp.c_loc = {-14.08115455, 9.62621962, -1.78361605, 0.08515207}; break;
      case 4:  // Be
        pp.z_ion = 4.0; pp.r_loc = 0.325;
        pp.c_loc = {-24.06746684, 17.27902186, -3.33910629, 0.16554912}; break;
      case 5:  // B
        pp.z_ion = 3.0; pp.r_loc = 0.41899145; pp.c_loc = {-5.85946171, 0.90375643, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.37132046, 6.29728018)}; break;
      case 6:  // C
        pp.z_ion = 4.0; pp.r_loc = 0.33847124; pp.c_loc = {-8.80367398, 1.33921085, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.30257575, 9.62248665)}; break;
      case 7:  // N
        pp.z_ion = 5.0; pp.r_loc = 0.28379051; pp.c_loc = {-12.41522559, 1.86809592, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.255405, 13.63026257)}; break;
      case 8:  // O
        pp.z_ion = 6.0; pp.r_loc = 0.2445543; pp.c_loc = {-16.6672148, 2.48731132, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.22095592, 18.33745811)}; break;
      case 9:  // F
        pp.z_ion = 7.0; pp.r_loc = 0.21492959; pp.c_loc = {-21.57302836, 3.19977615, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.19468402, 23.74354045)}; break;
      case 10:  // Ne
        pp.z_ion = 8.0; pp.r_loc = 0.19; pp.c_loc = {-27.12015973, 4.36044962, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.17605938, 28.17737082, 0.83365601, -1.07624528),
                          make_nl_1(1, 0.19547452, -0.2362936)}; break;
      case 11:  // Na
        pp.z_ion = 9.0; pp.r_loc = 0.23652322; pp.c_loc = {0.29510499, -0.91388488, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.14356046, 34.60149228),
                          make_nl_1(1, 0.12993224, -14.27746168)}; break;
      case 12:  // Mg
        pp.z_ion = 10.0; pp.r_loc = 0.19275787; pp.c_loc = {-20.57539077, 3.04016732, 0.0, 0.0};
        pp.nl_channels = {make_nl_1(0, 0.14140682, 41.04729209),
                          make_nl_1(1, 0.10293187, -9.98562566)}; break;
      case 13:  // Al
        pp.z_ion = 3.0; pp.r_loc = 0.45; pp.c_loc = {-7.55476126, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.48743529, 6.95993832, -1.88883584, 2.43847659),
                          make_nl_1(1, 0.56218949, 1.86529857)}; break;
      case 14:  // Si
        pp.z_ion = 4.0; pp.r_loc = 0.44; pp.c_loc = {-6.26928833, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.43563383, 8.9517415, -2.70627082, 3.4937806),
                          make_nl_1(1, 0.49794218, 2.43127673)}; break;
      case 15:  // P
        pp.z_ion = 5.0; pp.r_loc = 0.43; pp.c_loc = {-5.87594327, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.39637742, 11.00886207, -3.47035607, 4.48021042),
                          make_nl_1(1, 0.44829838, 3.05606416)}; break;
      case 16:  // S
        pp.z_ion = 6.0; pp.r_loc = 0.42; pp.c_loc = {-5.98626038, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.36482035, 13.14354448, -4.24183045, 5.47617957),
                          make_nl_1(1, 0.40948048, 3.70089057)}; break;
      case 17:  // Cl
        pp.z_ion = 7.0; pp.r_loc = 0.41; pp.c_loc = {-6.39208181, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.33953864, 15.21898983, -4.93452321, 6.37044208),
                          make_nl_1(1, 0.37847416, 4.33877527)}; break;
      case 18:  // Ar
        pp.z_ion = 8.0; pp.r_loc = 0.4; pp.c_loc = {-7.1, 0.0, 0.0, 0.0};
        pp.nl_channels = {make_nl_2(0, 0.31881468, 17.25203807, -5.58548836, 7.21083447),
                          make_nl_1(1, 0.35337019, 4.97421551)}; break;
    }
  }
  return pp;
}

GthPseudopotential lookup_gth_pseudopotential(std::string_view symbol, std::string_view family) {
  return lookup_gth_pseudopotential(element_atomic_number(symbol), family);
}

// ============================================================================
// PwCell: Lattice, Reciprocal Lattice, Structure Factor, Ewald & Madelung
// ============================================================================

namespace {

std::array<double, 3> cross3(const std::array<double, 3>& u, const std::array<double, 3>& v) {
  return {
      u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2],
      u[0] * v[1] - u[1] * v[0],
  };
}

double dot3(const std::array<double, 3>& u, const std::array<double, 3>& v) {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

double norm3(const std::array<double, 3>& u) {
  return std::sqrt(dot3(u, u));
}

}  // namespace

PwCell::PwCell(const std::array<std::array<double, 3>, 3>& a_vecs,
               std::vector<PwAtom> atoms,
               std::string_view pseudo_family)
    : a_vecs_(a_vecs), atoms_(std::move(atoms)), pseudo_family_(std::string(pseudo_family)) {
  const auto c12 = cross3(a_vecs_[1], a_vecs_[2]);
  const double det = dot3(a_vecs_[0], c12);
  if (std::abs(det) < 1e-12) {
    throw std::invalid_argument("PwCell: lattice vectors are singular or nearly coplanar");
  }
  volume_ = std::abs(det);

  // b_i = 2*pi * (a_j x a_k) / det  (signed det so a_i . b_j = 2*pi * delta_{ij})
  const auto c20 = cross3(a_vecs_[2], a_vecs_[0]);
  const auto c01 = cross3(a_vecs_[0], a_vecs_[1]);
  const double scale = kTwoPi / det;
  for (int d = 0; d < 3; ++d) {
    b_vecs_[0][d] = scale * c12[d];
    b_vecs_[1][d] = scale * c20[d];
    b_vecs_[2][d] = scale * c01[d];
  }

  const std::string fam_low = to_lower_trimmed(pseudo_family_);
  const bool bare_coulomb = (fam_low == "none" || fam_low == "bare" || fam_low == "all-electron");

  for (auto& atm : atoms_) {
    if (atm.atomic_number <= 0 && !atm.symbol.empty()) {
      atm.atomic_number = element_atomic_number(atm.symbol);
    } else if (atm.symbol.empty() && atm.atomic_number > 0 && atm.atomic_number <= 18) {
      atm.symbol = element_symbol(atm.atomic_number);
    }
    if (atm.pseudo.z_ion <= 0.0) {
      if (bare_coulomb) {
        atm.pseudo.atomic_number = atm.atomic_number;
        atm.pseudo.symbol = atm.symbol;
        atm.pseudo.z_ion = static_cast<double>(atm.atomic_number);
        atm.pseudo.r_loc = 0.0;
        atm.pseudo.c_loc = {0.0, 0.0, 0.0, 0.0};
      } else {
        atm.pseudo = lookup_gth_pseudopotential(atm.atomic_number, pseudo_family_);
      }
    }
  }
}

double PwCell::total_valence_charge() const noexcept {
  double z_sum = 0.0;
  for (const auto& atm : atoms_) z_sum += atm.pseudo.z_ion;
  return z_sum;
}

int PwCell::num_electrons(int charge) const {
  const double n_val = total_valence_charge() - static_cast<double>(charge);
  const int n_elec = static_cast<int>(std::llround(n_val));
  if (n_elec < 0 || std::abs(n_val - static_cast<double>(n_elec)) > 1e-6) {
    throw std::invalid_argument("PwCell::num_electrons: non-integer or negative electron count");
  }
  return n_elec;
}

std::vector<Complex> PwCell::structure_factor(int atom_index,
                                              std::span<const double> g_vecs_xyz) const {
  if (atom_index < 0 || atom_index >= natom()) {
    throw std::out_of_range("PwCell::structure_factor: atom_index out of range");
  }
  if (g_vecs_xyz.size() % 3 != 0) {
    throw std::invalid_argument("PwCell::structure_factor: g_vecs_xyz length must be 3*nG");
  }
  const std::size_t ng = g_vecs_xyz.size() / 3;
  const auto& r = atoms_[static_cast<std::size_t>(atom_index)].pos;
  std::vector<Complex> sf(ng);
  for (std::size_t ig = 0; ig < ng; ++ig) {
    const double phase = -(g_vecs_xyz[3 * ig + 0] * r[0] +
                           g_vecs_xyz[3 * ig + 1] * r[1] +
                           g_vecs_xyz[3 * ig + 2] * r[2]);
    sf[ig] = Complex(std::cos(phase), std::sin(phase));
  }
  return sf;
}

std::vector<double> PwCell::uniform_grid_coords(std::array<int, 3> mesh) const {
  if (mesh[0] <= 0 || mesh[1] <= 0 || mesh[2] <= 0) {
    throw std::invalid_argument("PwCell::uniform_grid_coords: mesh dimensions must be positive");
  }
  const std::size_t ntot =
      static_cast<std::size_t>(mesh[0]) * static_cast<std::size_t>(mesh[1]) * static_cast<std::size_t>(mesh[2]);
  std::vector<double> coords(ntot * 3);
  std::size_t idx = 0;
  for (int i0 = 0; i0 < mesh[0]; ++i0) {
    const double f0 = static_cast<double>(i0) / static_cast<double>(mesh[0]);
    for (int i1 = 0; i1 < mesh[1]; ++i1) {
      const double f1 = static_cast<double>(i1) / static_cast<double>(mesh[1]);
      for (int i2 = 0; i2 < mesh[2]; ++i2) {
        const double f2 = static_cast<double>(i2) / static_cast<double>(mesh[2]);
        for (int d = 0; d < 3; ++d) {
          coords[3 * idx + d] = f0 * a_vecs_[0][d] + f1 * a_vecs_[1][d] + f2 * a_vecs_[2][d];
        }
        ++idx;
      }
    }
  }
  return coords;
}

std::pair<std::vector<std::array<double, 3>>, std::vector<double>>
PwCell::monkhorst_pack_kpoints(std::array<int, 3> kmesh,
                               bool gamma_centered,
                               bool wrap_around) const {
  if (kmesh[0] <= 0 || kmesh[1] <= 0 || kmesh[2] <= 0) {
    throw std::invalid_argument("PwCell::monkhorst_pack_kpoints: kmesh must be positive");
  }
  std::array<std::vector<double>, 3> s_axes;
  for (int ax = 0; ax < 3; ++ax) {
    const int n = kmesh[ax];
    s_axes[ax].resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      double s = gamma_centered ? (static_cast<double>(i) / static_cast<double>(n))
                                : ((static_cast<double>(i) + 0.5) / static_cast<double>(n) - 0.5);
      if (wrap_around && s >= 0.5 - 1e-14) s -= 1.0;
      s_axes[ax][static_cast<std::size_t>(i)] = s;
    }
  }

  const int nk = kmesh[0] * kmesh[1] * kmesh[2];
  std::vector<std::array<double, 3>> kpts;
  kpts.reserve(static_cast<std::size_t>(nk));
  for (double s0 : s_axes[0]) {
    for (double s1 : s_axes[1]) {
      for (double s2 : s_axes[2]) {
        std::array<double, 3> kc{};
        for (int d = 0; d < 3; ++d) {
          kc[d] = s0 * b_vecs_[0][d] + s1 * b_vecs_[1][d] + s2 * b_vecs_[2][d];
        }
        kpts.push_back(kc);
      }
    }
  }
  std::vector<double> weights(static_cast<std::size_t>(nk), 1.0 / static_cast<double>(nk));
  return {std::move(kpts), std::move(weights)};
}

double PwCell::ewald_energy(double precision) const {
  if (atoms_.empty()) return 0.0;

  double z_sum = 0.0;
  double z2_sum = 0.0;
  for (const auto& atm : atoms_) {
    const double z = atm.pseudo.z_ion;
    z_sum += z;
    z2_sum += z * z;
  }
  if (z2_sum == 0.0) return 0.0;

  const double eta = std::sqrt(kPi) / std::cbrt(volume_);
  const double safe_prec = std::clamp(precision, 1e-16, 1e-8);
  const double log_term = std::sqrt(-std::log(safe_prec) + 8.0);
  const double r_cut = log_term / eta + 2.0;
  const double g_cut = 2.0 * eta * log_term + 2.0;

  double max_pair_dist = 0.0;
  for (std::size_t i = 0; i < atoms_.size(); ++i) {
    for (std::size_t j = i + 1; j < atoms_.size(); ++j) {
      std::array<double, 3> d{
          atoms_[i].pos[0] - atoms_[j].pos[0],
          atoms_[i].pos[1] - atoms_[j].pos[1],
          atoms_[i].pos[2] - atoms_[j].pos[2],
      };
      max_pair_dist = std::max(max_pair_dist, norm3(d));
    }
  }

  // Perpendicular cell heights h_alpha = 2*pi / |b_alpha| bound real-space lattice shells
  std::array<int, 3> n_real{};
  for (int ax = 0; ax < 3; ++ax) {
    const double h_ax = kTwoPi / norm3(b_vecs_[ax]);
    n_real[ax] = static_cast<int>(std::ceil((r_cut + max_pair_dist) / h_ax)) + 1;
  }

  double e_real = 0.0;
  const double r_cut2 = r_cut * r_cut;
  for (int l0 = -n_real[0]; l0 <= n_real[0]; ++l0) {
    for (int l1 = -n_real[1]; l1 <= n_real[1]; ++l1) {
      for (int l2 = -n_real[2]; l2 <= n_real[2]; ++l2) {
        const std::array<double, 3> L{
            l0 * a_vecs_[0][0] + l1 * a_vecs_[1][0] + l2 * a_vecs_[2][0],
            l0 * a_vecs_[0][1] + l1 * a_vecs_[1][1] + l2 * a_vecs_[2][1],
            l0 * a_vecs_[0][2] + l1 * a_vecs_[1][2] + l2 * a_vecs_[2][2],
        };
        for (std::size_t i = 0; i < atoms_.size(); ++i) {
          const double zi = atoms_[i].pseudo.z_ion;
          for (std::size_t j = 0; j < atoms_.size(); ++j) {
            if (l0 == 0 && l1 == 0 && l2 == 0 && i == j) continue;
            const double dx = atoms_[i].pos[0] - atoms_[j].pos[0] + L[0];
            const double dy = atoms_[i].pos[1] - atoms_[j].pos[1] + L[1];
            const double dz = atoms_[i].pos[2] - atoms_[j].pos[2] + L[2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 > r_cut2 || r2 < 1e-24) continue;
            const double r = std::sqrt(r2);
            e_real += 0.5 * zi * atoms_[j].pseudo.z_ion * std::erfc(eta * r) / r;
          }
        }
      }
    }
  }

  // Self-interaction + uniform neutralizing background
  const double e_self =
      -(eta / std::sqrt(kPi)) * z2_sum - 0.5 * kPi * (z_sum * z_sum) / (eta * eta * volume_);

  // Reciprocal-space Gaussian sum
  std::array<int, 3> m_recip{};
  for (int ax = 0; ax < 3; ++ax) {
    m_recip[ax] = static_cast<int>(std::ceil(norm3(a_vecs_[ax]) * g_cut / kTwoPi)) + 1;
  }

  double e_recip = 0.0;
  const double g_cut2 = g_cut * g_cut;
  const double inv_4eta2 = 0.25 / (eta * eta);
  const double pref_recip = kTwoPi / volume_;
  for (int m0 = -m_recip[0]; m0 <= m_recip[0]; ++m0) {
    for (int m1 = -m_recip[1]; m1 <= m_recip[1]; ++m1) {
      for (int m2 = -m_recip[2]; m2 <= m_recip[2]; ++m2) {
        if (m0 == 0 && m1 == 0 && m2 == 0) continue;
        const double gx = m0 * b_vecs_[0][0] + m1 * b_vecs_[1][0] + m2 * b_vecs_[2][0];
        const double gy = m0 * b_vecs_[0][1] + m1 * b_vecs_[1][1] + m2 * b_vecs_[2][1];
        const double gz = m0 * b_vecs_[0][2] + m1 * b_vecs_[1][2] + m2 * b_vecs_[2][2];
        const double g2 = gx * gx + gy * gy + gz * gz;
        if (g2 > g_cut2) continue;

        double s_re = 0.0;
        double s_im = 0.0;
        for (const auto& atm : atoms_) {
          const double phase = -(gx * atm.pos[0] + gy * atm.pos[1] + gz * atm.pos[2]);
          s_re += atm.pseudo.z_ion * std::cos(phase);
          s_im += atm.pseudo.z_ion * std::sin(phase);
        }
        const double s2 = s_re * s_re + s_im * s_im;
        e_recip += pref_recip * (std::exp(-g2 * inv_4eta2) / g2) * s2;
      }
    }
  }

  return e_real + e_self + e_recip;
}

double PwCell::madelung_constant(std::array<int, 3> kmesh, double precision) const {
  if (kmesh[0] <= 0 || kmesh[1] <= 0 || kmesh[2] <= 0) {
    throw std::invalid_argument("PwCell::madelung_constant: kmesh dimensions must be positive");
  }
  std::array<std::array<double, 3>, 3> super_a = a_vecs_;
  for (int ax = 0; ax < 3; ++ax) {
    for (int d = 0; d < 3; ++d) {
      super_a[ax][d] *= static_cast<double>(kmesh[ax]);
    }
  }
  PwAtom probe{};
  probe.atomic_number = 1;
  probe.symbol = "H";
  probe.pos = {0.0, 0.0, 0.0};
  probe.pseudo.z_ion = 1.0;
  PwCell super_cell(super_a, {probe}, "none");
  return -2.0 * super_cell.ewald_energy(precision);
}

// ============================================================================
// Self-Contained Host 1D & 3D Mixed-Radix (2,3,5) + Bluestein FFT
// ============================================================================

int next_fft_smooth_size(int n) noexcept {
  if (n <= 1) return 1;
  int cand = n;
  while (true) {
    int rem = cand;
    while (rem % 2 == 0) rem /= 2;
    while (rem % 3 == 0) rem /= 3;
    while (rem % 5 == 0) rem /= 5;
    if (rem == 1) return cand;
    ++cand;
  }
}

namespace {

void fft_pow2_inplace(Complex* a, int n, int direction) {
  int j = 0;
  for (int i = 1; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(a[i], a[j]);
  }
  const double sign = (direction < 0) ? -1.0 : 1.0;
  for (int len = 2; len <= n; len <<= 1) {
    const int half = len >> 1;
    const double ang = sign * kTwoPi / static_cast<double>(len);
    const Complex wlen(std::cos(ang), std::sin(ang));
    for (int i = 0; i < n; i += len) {
      Complex w(1.0, 0.0);
      for (int k = 0; k < half; ++k) {
        const Complex u = a[i + k];
        const Complex v = a[i + k + half] * w;
        a[i + k] = u + v;
        a[i + k + half] = u - v;
        w *= wlen;
      }
    }
  }
}

class Fft1DPlan {
 public:
  explicit Fft1DPlan(int n) : n_(n) {
    if (n <= 0) throw std::invalid_argument("Fft1DPlan: length must be positive");
    int rem = n;
    for (int p : {5, 3, 2}) {
      while (rem % p == 0) {
        factors_.push_back(p);
        rem /= p;
      }
    }
    smooth_ = (rem == 1);
    if (!smooth_) {
      // Bluestein Chirp-Z initialization for arbitrary prime factors
      m_pow2_ = 1;
      while (m_pow2_ < 2 * n_ - 1) m_pow2_ <<= 1;
      chirp_neg_.resize(static_cast<std::size_t>(n_));
      b_hat_neg_.assign(static_cast<std::size_t>(m_pow2_), Complex(0.0, 0.0));
      for (int k = 0; k < n_; ++k) {
        const std::int64_t k2_mod = (static_cast<std::int64_t>(k) * k) % (2 * n_);
        const double ang = -kPi * static_cast<double>(k2_mod) / static_cast<double>(n_);
        const Complex c(std::cos(ang), std::sin(ang));
        chirp_neg_[static_cast<std::size_t>(k)] = c;
        b_hat_neg_[static_cast<std::size_t>(k)] = std::conj(c);
        if (k > 0) {
          b_hat_neg_[static_cast<std::size_t>(m_pow2_ - k)] = std::conj(c);
        }
      }
      fft_pow2_inplace(b_hat_neg_.data(), m_pow2_, -1);
    }
  }

  void execute(const Complex* in, Complex* out, int direction, std::vector<Complex>& scratch) const {
    if (n_ == 1) {
      out[0] = in[0];
      return;
    }
    if (smooth_) {
      execute_stockham(in, out, direction, scratch);
    } else {
      execute_bluestein(in, out, direction, scratch);
    }
  }

 private:
  void execute_stockham(const Complex* in,
                        Complex* out,
                        int direction,
                        std::vector<Complex>& scratch) const {
    scratch.resize(static_cast<std::size_t>(2 * n_));
    Complex* src = scratch.data();
    Complex* dst = scratch.data() + n_;
    std::copy(in, in + n_, src);

    const double sign = (direction < 0) ? -1.0 : 1.0;
    const Complex w3_1(-0.5, sign * 0.86602540378443864676);
    const Complex w3_2(-0.5, -sign * 0.86602540378443864676);
    const double c72 = 0.30901699437494742410;
    const double s72 = sign * 0.95105651629515357212;
    const double c144 = -0.80901699437494742410;
    const double s144 = sign * 0.58778525229247312917;

    int m = 1;
    for (int r : factors_) {
      const int q = n_ / (r * m);
      const double base_ang = sign * kTwoPi / static_cast<double>(r * m);
      for (int k = 0; k < m; ++k) {
        Complex tw1(1.0, 0.0), tw2(1.0, 0.0), tw3(1.0, 0.0), tw4(1.0, 0.0);
        if (k > 0) {
          const double a1 = base_ang * static_cast<double>(k);
          tw1 = Complex(std::cos(a1), std::sin(a1));
          if (r >= 3) tw2 = tw1 * tw1;
          if (r == 5) {
            tw3 = tw2 * tw1;
            tw4 = tw2 * tw2;
          }
        }
        for (int j = 0; j < q; ++j) {
          if (r == 2) {
            const Complex x0 = src[k + m * (j + q * 0)];
            const Complex x1 = tw1 * src[k + m * (j + q * 1)];
            dst[k + m * (0 + 2 * j)] = x0 + x1;
            dst[k + m * (1 + 2 * j)] = x0 - x1;
          } else if (r == 3) {
            const Complex x0 = src[k + m * (j + q * 0)];
            const Complex x1 = tw1 * src[k + m * (j + q * 1)];
            const Complex x2 = tw2 * src[k + m * (j + q * 2)];
            dst[k + m * (0 + 3 * j)] = x0 + x1 + x2;
            dst[k + m * (1 + 3 * j)] = x0 + w3_1 * x1 + w3_2 * x2;
            dst[k + m * (2 + 3 * j)] = x0 + w3_2 * x1 + w3_1 * x2;
          } else {  // r == 5
            const Complex x0 = src[k + m * (j + q * 0)];
            const Complex x1 = tw1 * src[k + m * (j + q * 1)];
            const Complex x2 = tw2 * src[k + m * (j + q * 2)];
            const Complex x3 = tw3 * src[k + m * (j + q * 3)];
            const Complex x4 = tw4 * src[k + m * (j + q * 4)];
            const Complex s1 = x1 + x4;
            const Complex d1 = x1 - x4;
            const Complex s2 = x2 + x3;
            const Complex d2 = x2 - x3;
            dst[k + m * (0 + 5 * j)] = x0 + s1 + s2;
            const Complex r1 = x0 + c72 * s1 + c144 * s2;
            const Complex i1 = Complex(0.0, 1.0) * (s72 * d1 + s144 * d2);
            const Complex r2 = x0 + c144 * s1 + c72 * s2;
            const Complex i2 = Complex(0.0, 1.0) * (s144 * d1 - s72 * d2);
            dst[k + m * (1 + 5 * j)] = r1 + i1;
            dst[k + m * (4 + 5 * j)] = r1 - i1;
            dst[k + m * (2 + 5 * j)] = r2 + i2;
            dst[k + m * (3 + 5 * j)] = r2 - i2;
          }
        }
      }
      std::swap(src, dst);
      m *= r;
    }
    std::copy(src, src + n_, out);
  }

  void execute_bluestein(const Complex* in,
                         Complex* out,
                         int direction,
                         std::vector<Complex>& scratch) const {
    // Inverse DFT(x) = conj(Forward_DFT(conj(x)))
    scratch.assign(static_cast<std::size_t>(m_pow2_), Complex(0.0, 0.0));
    const bool inv = (direction > 0);
    for (int k = 0; k < n_; ++k) {
      const Complex xk = inv ? std::conj(in[k]) : in[k];
      scratch[static_cast<std::size_t>(k)] = xk * chirp_neg_[static_cast<std::size_t>(k)];
    }
    fft_pow2_inplace(scratch.data(), m_pow2_, -1);
    for (int k = 0; k < m_pow2_; ++k) {
      scratch[static_cast<std::size_t>(k)] *= b_hat_neg_[static_cast<std::size_t>(k)];
    }
    fft_pow2_inplace(scratch.data(), m_pow2_, +1);
    const double inv_m = 1.0 / static_cast<double>(m_pow2_);
    for (int k = 0; k < n_; ++k) {
      const Complex val =
          scratch[static_cast<std::size_t>(k)] * chirp_neg_[static_cast<std::size_t>(k)] * inv_m;
      out[k] = inv ? std::conj(val) : val;
    }
  }

  int n_ = 1;
  bool smooth_ = true;
  std::vector<int> factors_;
  int m_pow2_ = 1;
  std::vector<Complex> chirp_neg_;
  std::vector<Complex> b_hat_neg_;
};

void fft3d_batch_host(std::array<int, 3> grid,
                      int nbatch,
                      const Complex* in,
                      Complex* out,
                      int direction) {
  const int n0 = grid[0];
  const int n1 = grid[1];
  const int n2 = grid[2];
  if (n0 <= 0 || n1 <= 0 || n2 <= 0 || nbatch <= 0) return;

  const std::size_t ngrid =
      static_cast<std::size_t>(n0) * static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2);
  if (in != out) {
    std::copy(in, in + static_cast<std::size_t>(nbatch) * ngrid, out);
  }

  const Fft1DPlan plan0(n0);
  const Fft1DPlan plan1(n1);
  const Fft1DPlan plan2(n2);

  // Axis 2 (contiguous length n2)
  const std::ptrdiff_t lines2 = static_cast<std::ptrdiff_t>(nbatch) * n0 * n1;
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<Complex> scratch;
    std::vector<Complex> line_out(static_cast<std::size_t>(n2));
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t idx = 0; idx < lines2; ++idx) {
      Complex* base = out + static_cast<std::size_t>(idx) * static_cast<std::size_t>(n2);
      plan2.execute(base, line_out.data(), direction, scratch);
      std::copy(line_out.begin(), line_out.end(), base);
    }
  }

  // Axis 1 (stride n2, length n1)
  const std::ptrdiff_t lines1 = static_cast<std::ptrdiff_t>(nbatch) * n0 * n2;
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<Complex> scratch;
    std::vector<Complex> line_in(static_cast<std::size_t>(n1));
    std::vector<Complex> line_out(static_cast<std::size_t>(n1));
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t idx = 0; idx < lines1; ++idx) {
      const std::size_t i2 = static_cast<std::size_t>(idx) % static_cast<std::size_t>(n2);
      const std::size_t bi0 = static_cast<std::size_t>(idx) / static_cast<std::size_t>(n2);
      Complex* plane = out + bi0 * static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2) + i2;
      for (int i1 = 0; i1 < n1; ++i1) {
        line_in[static_cast<std::size_t>(i1)] = plane[static_cast<std::size_t>(i1) * n2];
      }
      plan1.execute(line_in.data(), line_out.data(), direction, scratch);
      for (int i1 = 0; i1 < n1; ++i1) {
        plane[static_cast<std::size_t>(i1) * n2] = line_out[static_cast<std::size_t>(i1)];
      }
    }
  }

  // Axis 0 (stride n1*n2, length n0)
  const std::size_t stride0 = static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2);
  const std::ptrdiff_t lines0 = static_cast<std::ptrdiff_t>(nbatch) * static_cast<std::ptrdiff_t>(stride0);
#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<Complex> scratch;
    std::vector<Complex> line_in(static_cast<std::size_t>(n0));
    std::vector<Complex> line_out(static_cast<std::size_t>(n0));
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t idx = 0; idx < lines0; ++idx) {
      const std::size_t rem = static_cast<std::size_t>(idx) % stride0;
      const std::size_t b = static_cast<std::size_t>(idx) / stride0;
      Complex* cube = out + b * ngrid + rem;
      for (int i0 = 0; i0 < n0; ++i0) {
        line_in[static_cast<std::size_t>(i0)] = cube[static_cast<std::size_t>(i0) * stride0];
      }
      plan0.execute(line_in.data(), line_out.data(), direction, scratch);
      for (int i0 = 0; i0 < n0; ++i0) {
        cube[static_cast<std::size_t>(i0) * stride0] = line_out[static_cast<std::size_t>(i0)];
      }
    }
  }
}

}  // namespace

bool cuda_pw_available() noexcept {
#if defined(GPU4DFT_HAVE_CUDA)
  return detail::pw_cuda_device_count() > 0;
#else
  return false;
#endif
}

void fft3d_batch(std::array<int, 3> grid,
                 int nbatch,
                 const Complex* in,
                 Complex* out,
                 int direction,
                 Backend backend) {
  if (direction != -1 && direction != 1) {
    throw std::invalid_argument("fft3d_batch: direction must be -1 (FORWARD) or +1 (INVERSE)");
  }
  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::pw_cuda_device_count() == 0) {
      throw std::runtime_error("fft3d_batch: no CUDA device available");
    }
    detail::pw_cuda_fft3d_batch(grid, nbatch, in, out, direction);
    return;
#else
    throw std::runtime_error("fft3d_batch: gpu4dft was built without CUDA");
#endif
  }
  if (backend != Backend::Host) {
    throw std::runtime_error("fft3d_batch: unsupported backend");
  }
  fft3d_batch_host(grid, nbatch, in, out, direction);
}

// ============================================================================
// PwBasis & NonLocalProjector
// ============================================================================

PwBasis::PwBasis(const PwCell& cell,
                 double ecut_wfc,
                 double ecut_rho,
                 std::span<const std::array<double, 3>> kpoints,
                 std::span<const double> kweights,
                 std::array<int, 3> fft_grid_override)
    : cell_(cell),
      ecut_wfc_(ecut_wfc),
      ecut_rho_((ecut_rho > 0.0) ? ecut_rho : (4.0 * ecut_wfc)) {
  if (ecut_wfc_ <= 0.0) {
    throw std::invalid_argument("PwBasis: ecut_wfc must be positive");
  }
  const auto& a = cell_.lattice_vectors();
  const auto& b = cell_.reciprocal_vectors();
  const double gmax_rho = std::sqrt(2.0 * ecut_rho_);

  for (int ax = 0; ax < 3; ++ax) {
    if (fft_grid_override[ax] > 0) {
      fft_grid_[ax] = fft_grid_override[ax];
    } else {
      const int gmax_ax = static_cast<int>(std::ceil(norm3(a[ax]) * gmax_rho / kTwoPi - 1e-12));
      fft_grid_[ax] = next_fft_smooth_size(2 * gmax_ax + 1);
    }
  }

  ngrid_ = fft_grid_[0] * fft_grid_[1] * fft_grid_[2];
  g_vecs_xyz_.resize(static_cast<std::size_t>(ngrid_) * 3);
  g2_.resize(static_cast<std::size_t>(ngrid_));

  std::size_t idx = 0;
  for (int j0 = 0; j0 < fft_grid_[0]; ++j0) {
    const int m0 = (j0 <= (fft_grid_[0] - 1) / 2) ? j0 : (j0 - fft_grid_[0]);
    for (int j1 = 0; j1 < fft_grid_[1]; ++j1) {
      const int m1 = (j1 <= (fft_grid_[1] - 1) / 2) ? j1 : (j1 - fft_grid_[1]);
      for (int j2 = 0; j2 < fft_grid_[2]; ++j2) {
        const int m2 = (j2 <= (fft_grid_[2] - 1) / 2) ? j2 : (j2 - fft_grid_[2]);
        const double gx = m0 * b[0][0] + m1 * b[1][0] + m2 * b[2][0];
        const double gy = m0 * b[0][1] + m1 * b[1][1] + m2 * b[2][1];
        const double gz = m0 * b[0][2] + m1 * b[1][2] + m2 * b[2][2];
        g_vecs_xyz_[3 * idx + 0] = gx;
        g_vecs_xyz_[3 * idx + 1] = gy;
        g_vecs_xyz_[3 * idx + 2] = gz;
        g2_[idx] = gx * gx + gy * gy + gz * gz;
        ++idx;
      }
    }
  }

  std::vector<std::array<double, 3>> k_list;
  std::vector<double> w_list;
  if (kpoints.empty()) {
    k_list.push_back({0.0, 0.0, 0.0});
    w_list.push_back(1.0);
  } else {
    k_list.assign(kpoints.begin(), kpoints.end());
    if (kweights.empty()) {
      w_list.assign(k_list.size(), 1.0 / static_cast<double>(k_list.size()));
    } else {
      if (kweights.size() != k_list.size()) {
        throw std::invalid_argument("PwBasis: kweights size mismatch");
      }
      w_list.assign(kweights.begin(), kweights.end());
    }
  }

  kpts_.resize(k_list.size());
  for (std::size_t ik = 0; ik < k_list.size(); ++ik) {
    auto& kp = kpts_[ik];
    kp.k_cart = k_list[ik];
    kp.weight = w_list[ik];
    for (int ig = 0; ig < ngrid_; ++ig) {
      const std::size_t uig = static_cast<std::size_t>(ig);
      const double gx = g_vecs_xyz_[3 * uig + 0];
      const double gy = g_vecs_xyz_[3 * uig + 1];
      const double gz = g_vecs_xyz_[3 * uig + 2];
      const double qx = kp.k_cart[0] + gx;
      const double qy = kp.k_cart[1] + gy;
      const double qz = kp.k_cart[2] + gz;
      const double kin = 0.5 * (qx * qx + qy * qy + qz * qz);
      if (kin <= ecut_wfc_ + 1e-12) {
        kp.fft_indices.push_back(ig);
        kp.kin.push_back(kin);
        kp.kg_xyz.push_back(qx);
        kp.kg_xyz.push_back(qy);
        kp.kg_xyz.push_back(qz);
        kp.g_xyz.push_back(gx);
        kp.g_xyz.push_back(gy);
        kp.g_xyz.push_back(gz);
      }
    }
    kp.npw = static_cast<int>(kp.fft_indices.size());
    if (kp.npw == 0) {
      throw std::invalid_argument("PwBasis: zero active plane waves for given ecut_wfc");
    }
  }
}

std::vector<double> PwBasis::compute_local_pp_grid() const {
  const std::size_t ng = static_cast<std::size_t>(ngrid_);
  std::vector<Complex> vloc_g(ng, Complex(0.0, 0.0));
  const double inv_vol = 1.0 / cell_.volume();
  const double two_pi_32 = std::pow(kTwoPi, 1.5);

  for (const auto& atm : cell_.atoms()) {
    const auto& pp = atm.pseudo;
    const double zion = pp.z_ion;
    const double rloc = pp.r_loc;
    const double r2 = rloc * rloc;
    const double r3 = r2 * rloc;
    const double c1 = pp.c_loc[0];
    const double c2 = pp.c_loc[1];
    const double c3 = pp.c_loc[2];
    const double c4 = pp.c_loc[3];
    const double eps_core = pp.core_energy_constant();

    for (std::size_t ig = 0; ig < ng; ++ig) {
      const double g2 = g2_[ig];
      if (g2 < 1e-18) {
        vloc_g[ig] += Complex(inv_vol * eps_core, 0.0);
        continue;
      }
      const double x2 = g2 * r2;
      const double x4 = x2 * x2;
      const double x6 = x4 * x2;
      const double poly = c1 + c2 * (3.0 - x2) + c3 * (15.0 - 10.0 * x2 + x4) +
                          c4 * (105.0 - 105.0 * x2 + 21.0 * x4 - x6);
      const double vi =
          (-kFourPi * zion / g2 + two_pi_32 * r3 * poly) * std::exp(-0.5 * x2);
      const double phase = -(g_vecs_xyz_[3 * ig + 0] * atm.pos[0] +
                             g_vecs_xyz_[3 * ig + 1] * atm.pos[1] +
                             g_vecs_xyz_[3 * ig + 2] * atm.pos[2]);
      vloc_g[ig] += (inv_vol * vi) * Complex(std::cos(phase), std::sin(phase));
    }
  }

  std::vector<Complex> vloc_r_c(ng);
  fft3d_batch_host(fft_grid_, 1, vloc_g.data(), vloc_r_c.data(), +1);
  std::vector<double> vloc_r(ng);
  for (std::size_t ig = 0; ig < ng; ++ig) {
    vloc_r[ig] = vloc_r_c[ig].real();
  }
  return vloc_r;
}

std::vector<double> PwBasis::compute_exx_coulomb_kernel(const std::array<double, 3>& q_shift,
                                                        ExxDiv exxdiv,
                                                        std::array<int, 3> kmesh) const {
  const std::size_t ng = static_cast<std::size_t>(ngrid_);
  std::vector<double> coul_g(ng, 0.0);
  const auto& a = cell_.lattice_vectors();
  const auto& b = cell_.reciprocal_vectors();

  const double q_norm = norm3(q_shift);
  const bool has_shift = (q_norm > 1e-10);
  double g0_val = 0.0;
  if (exxdiv == ExxDiv::Ewald) {
    const double nk_prod = static_cast<double>(kmesh[0] * kmesh[1] * kmesh[2]);
    g0_val = nk_prod * cell_.volume() * cell_.madelung_constant(kmesh);
  }

  for (std::size_t ig = 0; ig < ng; ++ig) {
    double qx = q_shift[0] + g_vecs_xyz_[3 * ig + 0];
    double qy = q_shift[1] + g_vecs_xyz_[3 * ig + 1];
    double qz = q_shift[2] + g_vecs_xyz_[3 * ig + 2];

    if (has_shift) {
      // Minimum-image wrap-around in fractional reciprocal coordinates (matches PySCF _Gv_wrap_around)
      std::array<double, 3> s{};
      for (int ax = 0; ax < 3; ++ax) {
        s[ax] = (qx * a[ax][0] + qy * a[ax][1] + qz * a[ax][2]) / kTwoPi;
        const double half_n = 0.5 * static_cast<double>(fft_grid_[ax]);
        if (s[ax] > half_n + 1e-9) s[ax] -= static_cast<double>(fft_grid_[ax]);
        if (s[ax] < -half_n - 1e-9) s[ax] += static_cast<double>(fft_grid_[ax]);
      }
      qx = s[0] * b[0][0] + s[1] * b[1][0] + s[2] * b[2][0];
      qy = s[0] * b[0][1] + s[1] * b[1][1] + s[2] * b[2][1];
      qz = s[0] * b[0][2] + s[1] * b[1][2] + s[2] * b[2][2];
    }

    const double q2 = qx * qx + qy * qy + qz * qz;
    if (q2 < 1e-18) {
      coul_g[ig] = g0_val;
    } else {
      coul_g[ig] = kFourPi / q2;
    }
  }
  return coul_g;
}

NonLocalProjector::NonLocalProjector(const PwCell& cell, const KPointBasis& kp)
    : npw_(kp.npw), nproj_(0) {
  int total_proj = 0;
  for (const auto& atm : cell.atoms()) {
    for (const auto& ch : atm.pseudo.nl_channels) {
      if (ch.n_proj > 0) total_proj += (2 * ch.l + 1) * ch.n_proj;
    }
  }
  nproj_ = total_proj;
  if (nproj_ == 0) return;

  P_.assign(static_cast<std::size_t>(npw_) * static_cast<std::size_t>(nproj_), Complex(0.0, 0.0));
  H_.assign(static_cast<std::size_t>(nproj_) * static_cast<std::size_t>(nproj_), Complex(0.0, 0.0));

  const double inv_sqrt_vol = 1.0 / std::sqrt(cell.volume());
  const double pi_54 = std::pow(kPi, 1.25);

  int col_offset = 0;
  for (const auto& atm : cell.atoms()) {
    for (const auto& ch : atm.pseudo.nl_channels) {
      if (ch.n_proj <= 0) continue;
      const int l = ch.l;
      const int nm = 2 * l + 1;
      const int np = ch.n_proj;
      const double rl = ch.r_l;
      const double rl_pow = std::pow(rl, static_cast<double>(l) + 1.5);

      // (-i)^l phase factor
      Complex minus_i_pow_l(1.0, 0.0);
      if ((l & 3) == 1) minus_i_pow_l = Complex(0.0, -1.0);
      else if ((l & 3) == 2) minus_i_pow_l = Complex(-1.0, 0.0);
      else if ((l & 3) == 3) minus_i_pow_l = Complex(0.0, 1.0);

      for (int m = 0; m < nm; ++m) {
        const int base = col_offset + m * np;
        for (int i = 0; i < np; ++i) {
          for (int j = 0; j < np; ++j) {
            H_[static_cast<std::size_t>(base + i) +
               static_cast<std::size_t>(base + j) * static_cast<std::size_t>(nproj_)] =
                Complex(ch.h[static_cast<std::size_t>(i * np + j)], 0.0);
          }
        }
      }

      for (int ig = 0; ig < npw_; ++ig) {
        const std::size_t uig = static_cast<std::size_t>(ig);
        const double gx = kp.g_xyz[3 * uig + 0];
        const double gy = kp.g_xyz[3 * uig + 1];
        const double gz = kp.g_xyz[3 * uig + 2];
        const double qx = kp.kg_xyz[3 * uig + 0];
        const double qy = kp.kg_xyz[3 * uig + 1];
        const double qz = kp.kg_xyz[3 * uig + 2];
        const double q_norm = std::sqrt(qx * qx + qy * qy + qz * qz);
        const double qr = q_norm * rl;

        double slm[9];
        detail::eval_solid_harmonics_pw(l, qx, qy, qz, slm);

        const double phase = -(gx * atm.pos[0] + gy * atm.pos[1] + gz * atm.pos[2]);
        const Complex sf_phase = Complex(std::cos(phase), std::sin(phase)) * minus_i_pow_l;
        const double rad_pref = inv_sqrt_vol * pi_54 * rl_pow * std::exp(-0.5 * qr * qr);

        for (int m = 0; m < nm; ++m) {
          const double ang_val = rad_pref * slm[m];
          for (int i = 0; i < np; ++i) {
            const int col = col_offset + m * np + i;
            const double qli = detail::gth_qli(qr, l, i);
            P_[uig + static_cast<std::size_t>(col) * static_cast<std::size_t>(npw_)] =
                sf_phase * (ang_val * qli);
          }
        }
      }

      col_offset += nm * np;
    }
  }
}

// ============================================================================
// Self-Contained Host Complex Linear Algebra (ZGEMM, ZHEEV, ZPOTRF, SVQB)
// ============================================================================

namespace {

// Column-major C[m x n] = alpha * op(A)[m x k] * op(B)[k x n] + beta * C[m x n]
void zgemm_host(char opA,
                char opB,
                int m,
                int n,
                int k,
                Complex alpha,
                const Complex* A,
                int lda,
                const Complex* B,
                int ldb,
                Complex beta,
                Complex* C,
                int ldc) {
  if (m <= 0 || n <= 0) return;
  if (beta == Complex(0.0, 0.0)) {
    for (int j = 0; j < n; ++j) {
      std::fill(C + static_cast<std::size_t>(j) * ldc,
                C + static_cast<std::size_t>(j) * ldc + m,
                Complex(0.0, 0.0));
    }
  } else if (beta != Complex(1.0, 0.0)) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        C[i + static_cast<std::size_t>(j) * ldc] *= beta;
      }
    }
  }
  if (k <= 0 || alpha == Complex(0.0, 0.0)) return;

  const bool conjA = (opA == 'C' || opA == 'c');
  const bool conjB = (opB == 'C' || opB == 'c');

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (static_cast<long long>(m) * n * k > 16384)
#endif
  for (int j = 0; j < n; ++j) {
    Complex* c_col = C + static_cast<std::size_t>(j) * ldc;
    if (!conjA) {
      for (int p = 0; p < k; ++p) {
        const Complex b_pj =
            conjB ? std::conj(B[j + static_cast<std::size_t>(p) * ldb])
                  : B[p + static_cast<std::size_t>(j) * ldb];
        const Complex scale = alpha * b_pj;
        if (scale == Complex(0.0, 0.0)) continue;
        const Complex* a_col = A + static_cast<std::size_t>(p) * lda;
        for (int i = 0; i < m; ++i) {
          c_col[i] += a_col[i] * scale;
        }
      }
    } else {
      for (int i = 0; i < m; ++i) {
        const Complex* a_col_i = A + static_cast<std::size_t>(i) * lda;
        Complex sum(0.0, 0.0);
        if (!conjB) {
          const Complex* b_col_j = B + static_cast<std::size_t>(j) * ldb;
          for (int p = 0; p < k; ++p) {
            sum += std::conj(a_col_i[p]) * b_col_j[p];
          }
        } else {
          for (int p = 0; p < k; ++p) {
            sum += std::conj(a_col_i[p]) * std::conj(B[j + static_cast<std::size_t>(p) * ldb]);
          }
        }
        c_col[i] += alpha * sum;
      }
    }
  }
}

// Implicit QL with Wilkinson shift for real symmetric tridiagonal matrix (diag d[n], subdiag e[n]).
void steqr_ql_host(int n, double* d, double* e, double* z_col_major) {
  if (n <= 1) return;
  e[n - 1] = 0.0;

  for (int l = 0; l < n; ++l) {
    int iter = 0;
    while (true) {
      int m = l;
      for (; m < n - 1; ++m) {
        const double dd = std::abs(d[m]) + std::abs(d[m + 1]);
        if (std::abs(e[m]) <= 1e-15 * (dd + 1e-300)) break;
      }
      if (m == l) break;
      if (++iter > 60 * n) {
        throw std::runtime_error("steqr_ql_host: failed to converge");
      }

      double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
      double r = std::hypot(g, 1.0);
      g = d[m] - d[l] + e[l] / (g + ((g >= 0.0) ? r : -r));
      double s = 1.0;
      double c = 1.0;
      double p = 0.0;

      for (int i = m - 1; i >= l; --i) {
        const double f = s * e[i];
        const double b = c * e[i];
        r = std::hypot(f, g);
        e[i + 1] = r;
        if (r == 0.0) {
          d[i + 1] -= p;
          e[m] = 0.0;
          break;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2.0 * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b;

        double* zi = z_col_major + static_cast<std::size_t>(i) * n;
        double* zi1 = z_col_major + static_cast<std::size_t>(i + 1) * n;
        for (int k = 0; k < n; ++k) {
          const double t = zi1[k];
          zi1[k] = s * zi[k] + c * t;
          zi[k] = c * zi[k] - s * t;
        }
      }
      if (r == 0.0 && iter > 0) continue;
      d[l] -= p;
      e[l] = g;
      e[m] = 0.0;
    }
  }
}

// Complex Hermitian eigensolver A * U = U * diag(w) (column-major [n x n], ascending w).
// Uses complex Householder tridiagonalization + unit phase gauge -> real symmetric QL.
void zheev_host(int n, const Complex* A_in, double* w_out, Complex* U_out) {
  if (n <= 0) return;
  if (n == 1) {
    w_out[0] = A_in[0].real();
    U_out[0] = Complex(1.0, 0.0);
    return;
  }

  std::vector<Complex> H(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      H[i + static_cast<std::size_t>(j) * n] =
          0.5 * (A_in[i + static_cast<std::size_t>(j) * n] +
                 std::conj(A_in[j + static_cast<std::size_t>(i) * n]));
    }
    H[j + static_cast<std::size_t>(j) * n] =
        Complex(H[j + static_cast<std::size_t>(j) * n].real(), 0.0);
  }

  std::vector<Complex> Q(static_cast<std::size_t>(n) * n, Complex(0.0, 0.0));
  for (int i = 0; i < n; ++i) Q[i + static_cast<std::size_t>(i) * n] = Complex(1.0, 0.0);

  std::vector<Complex> v(static_cast<std::size_t>(n));
  std::vector<Complex> p_vec(static_cast<std::size_t>(n));
  std::vector<Complex> w_vec(static_cast<std::size_t>(n));

  for (int k = 0; k < n - 2; ++k) {
    double norm_x2 = 0.0;
    for (int i = k + 1; i < n; ++i) {
      norm_x2 += std::norm(H[i + static_cast<std::size_t>(k) * n]);
    }
    const double norm_x = std::sqrt(norm_x2);
    if (norm_x < 1e-15) continue;

    const Complex x1 = H[(k + 1) + static_cast<std::size_t>(k) * n];
    const double abs_x1 = std::abs(x1);
    const Complex phase = (abs_x1 > 1e-30) ? (x1 / abs_x1) : Complex(1.0, 0.0);
    const Complex alpha = -phase * norm_x;

    std::fill(v.begin(), v.end(), Complex(0.0, 0.0));
    v[k + 1] = x1 - alpha;
    for (int i = k + 2; i < n; ++i) {
      v[i] = H[i + static_cast<std::size_t>(k) * n];
    }
    double v_norm2 = 0.0;
    for (int i = k + 1; i < n; ++i) v_norm2 += std::norm(v[i]);
    if (v_norm2 < 1e-30) continue;
    const double inv_vnorm = 1.0 / std::sqrt(v_norm2);
    for (int i = k + 1; i < n; ++i) v[i] *= inv_vnorm;

    // Two-sided Hermitian reflection: H <- (I - 2 v v^H) H (I - 2 v v^H)
    for (int i = k + 1; i < n; ++i) {
      Complex sum(0.0, 0.0);
      for (int j = k + 1; j < n; ++j) {
        sum += H[i + static_cast<std::size_t>(j) * n] * v[j];
      }
      p_vec[i] = 2.0 * sum;
    }
    Complex vHp(0.0, 0.0);
    for (int i = k + 1; i < n; ++i) vHp += std::conj(v[i]) * p_vec[i];
    for (int i = k + 1; i < n; ++i) w_vec[i] = p_vec[i] - vHp * v[i];

    for (int j = k + 1; j < n; ++j) {
      for (int i = k + 1; i < n; ++i) {
        H[i + static_cast<std::size_t>(j) * n] -=
            v[i] * std::conj(w_vec[j]) + w_vec[i] * std::conj(v[j]);
      }
      H[j + static_cast<std::size_t>(j) * n] =
          Complex(H[j + static_cast<std::size_t>(j) * n].real(), 0.0);
    }
    H[(k + 1) + static_cast<std::size_t>(k) * n] = alpha;
    H[k + static_cast<std::size_t>(k + 1) * n] = std::conj(alpha);
    for (int i = k + 2; i < n; ++i) {
      H[i + static_cast<std::size_t>(k) * n] = Complex(0.0, 0.0);
      H[k + static_cast<std::size_t>(i) * n] = Complex(0.0, 0.0);
    }

    // Accumulate Q <- Q (I - 2 v v^H)
    for (int i = 0; i < n; ++i) {
      Complex qv(0.0, 0.0);
      for (int j = k + 1; j < n; ++j) {
        qv += Q[i + static_cast<std::size_t>(j) * n] * v[j];
      }
      const Complex two_qv = 2.0 * qv;
      for (int j = k + 1; j < n; ++j) {
        Q[i + static_cast<std::size_t>(j) * n] -= two_qv * std::conj(v[j]);
      }
    }
  }

  // Diagonal unitary phase gauge D so D^H H D has purely real non-negative subdiagonal e[k]
  std::vector<double> d(static_cast<std::size_t>(n));
  std::vector<double> e(static_cast<std::size_t>(n), 0.0);
  std::vector<Complex> phase_d(static_cast<std::size_t>(n), Complex(1.0, 0.0));
  d[0] = H[0].real();
  for (int k = 0; k < n - 1; ++k) {
    d[k + 1] = H[(k + 1) + static_cast<std::size_t>(k + 1) * n].real();
    const Complex sub = H[(k + 1) + static_cast<std::size_t>(k) * n] * phase_d[k];
    const double abs_sub = std::abs(sub);
    e[k] = abs_sub;
    phase_d[k + 1] = (abs_sub > 1e-30) ? (sub / abs_sub) : Complex(1.0, 0.0);
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      Q[i + static_cast<std::size_t>(j) * n] *= phase_d[j];
    }
  }

  std::vector<double> Z(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) Z[i + static_cast<std::size_t>(i) * n] = 1.0;
  steqr_ql_host(n, d.data(), e.data(), Z.data());

  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return d[a] < d[b]; });

  for (int col = 0; col < n; ++col) {
    const int src_col = order[static_cast<std::size_t>(col)];
    w_out[col] = d[static_cast<std::size_t>(src_col)];
    for (int i = 0; i < n; ++i) {
      Complex sum(0.0, 0.0);
      for (int k = 0; k < n; ++k) {
        sum += Q[i + static_cast<std::size_t>(k) * n] * Z[k + static_cast<std::size_t>(src_col) * n];
      }
      U_out[i + static_cast<std::size_t>(col) * n] = sum;
    }
  }
}

// Cholesky B = L * L^H for Hermitian positive-definite column-major B [n x n],
// followed by solving Xi * L^H = W (i.e. Xi = W * L^{-H}) for column-major W [m x n].
void cholesky_and_trsm_ace_host(int m, int n, const Complex* B_in, const Complex* W_in, Complex* Xi_out) {
  std::vector<Complex> L(static_cast<std::size_t>(n) * n, Complex(0.0, 0.0));
  for (int j = 0; j < n; ++j) {
    double diag = 0.5 * (B_in[j + static_cast<std::size_t>(j) * n].real() +
                         B_in[j + static_cast<std::size_t>(j) * n].real());
    for (int k = 0; k < j; ++k) {
      diag -= std::norm(L[j + static_cast<std::size_t>(k) * n]);
    }
    if (diag <= 1e-14) {
      // Fallback via Hermitian eigendecomposition B^{-1/2} if semi-definite
      std::vector<double> evals(static_cast<std::size_t>(n));
      std::vector<Complex> evecs(static_cast<std::size_t>(n) * n);
      zheev_host(n, B_in, evals.data(), evecs.data());
      std::vector<Complex> inv_sqrt_mat(static_cast<std::size_t>(n) * n, Complex(0.0, 0.0));
      for (int k = 0; k < n; ++k) {
        if (evals[k] > 1e-13) {
          const double s = 1.0 / std::sqrt(evals[k]);
          for (int r = 0; r < n; ++r) {
            inv_sqrt_mat[r + static_cast<std::size_t>(k) * n] =
                evecs[r + static_cast<std::size_t>(k) * n] * s;
          }
        }
      }
      zgemm_host('N', 'N', m, n, n, Complex(1.0, 0.0), W_in, m, inv_sqrt_mat.data(), n,
                 Complex(0.0, 0.0), Xi_out, m);
      return;
    }
    const double l_jj = std::sqrt(diag);
    L[j + static_cast<std::size_t>(j) * n] = Complex(l_jj, 0.0);
    const double inv_ljj = 1.0 / l_jj;
    for (int i = j + 1; i < n; ++i) {
      Complex sum = 0.5 * (B_in[i + static_cast<std::size_t>(j) * n] +
                           std::conj(B_in[j + static_cast<std::size_t>(i) * n]));
      for (int k = 0; k < j; ++k) {
        sum -= L[i + static_cast<std::size_t>(k) * n] *
               std::conj(L[j + static_cast<std::size_t>(k) * n]);
      }
      L[i + static_cast<std::size_t>(j) * n] = sum * inv_ljj;
    }
  }

  // Forward substitution on columns j = 0..n-1 of Xi: Xi[:, j] = (W[:, j] - sum_{k<j} Xi[:, k] * conj(L[j, k])) / L[j, j]
  std::copy(W_in, W_in + static_cast<std::size_t>(m) * n, Xi_out);
  for (int j = 0; j < n; ++j) {
    Complex* xi_j = Xi_out + static_cast<std::size_t>(j) * m;
    for (int k = 0; k < j; ++k) {
      const Complex l_jk_conj = std::conj(L[j + static_cast<std::size_t>(k) * n]);
      const Complex* xi_k = Xi_out + static_cast<std::size_t>(k) * m;
      for (int r = 0; r < m; ++r) {
        xi_j[r] -= xi_k[r] * l_jk_conj;
      }
    }
    const double inv_ljj = 1.0 / L[j + static_cast<std::size_t>(j) * n].real();
    for (int r = 0; r < m; ++r) xi_j[r] *= inv_ljj;
  }
}

// Kahan two-pass orthogonalization against V_against [npw x n_against] + SVQB orthonormalization.
// Optionally applies the exact same column transformations to HV_sync!
int svqb_orthonormalize_host(int npw,
                             Complex* V,
                             int ncols,
                             const Complex* V_against,
                             int n_against,
                             Complex* HV_sync = nullptr,
                             const Complex* HV_against = nullptr) {
  if (ncols <= 0) return 0;

  if (V_against != nullptr && n_against > 0) {
    std::vector<Complex> C(static_cast<std::size_t>(n_against) * ncols);
    for (int pass = 0; pass < 2; ++pass) {
      zgemm_host('C', 'N', n_against, ncols, npw, Complex(1.0, 0.0), V_against, npw, V, npw,
                 Complex(0.0, 0.0), C.data(), n_against);
      zgemm_host('N', 'N', npw, ncols, n_against, Complex(-1.0, 0.0), V_against, npw, C.data(),
                 n_against, Complex(1.0, 0.0), V, npw);
      if (HV_sync != nullptr && HV_against != nullptr) {
        zgemm_host('N', 'N', npw, ncols, n_against, Complex(-1.0, 0.0), HV_against, npw, C.data(),
                   n_against, Complex(1.0, 0.0), HV_sync, npw);
      }
    }
  }

  // Normalize columns with non-trivial norm first so S = V^H V is well-scaled
  int n_keep_pre = 0;
  for (int j = 0; j < ncols; ++j) {
    double nrm2 = 0.0;
    const Complex* col = V + static_cast<std::size_t>(j) * npw;
    for (int i = 0; i < npw; ++i) nrm2 += std::norm(col[i]);
    const double nrm = std::sqrt(nrm2);
    if (nrm > 1e-13) {
      const double inv_nrm = 1.0 / nrm;
      Complex* dst = V + static_cast<std::size_t>(n_keep_pre) * npw;
      for (int i = 0; i < npw; ++i) dst[i] = col[i] * inv_nrm;
      if (HV_sync != nullptr) {
        const Complex* hsrc = HV_sync + static_cast<std::size_t>(j) * npw;
        Complex* hdst = HV_sync + static_cast<std::size_t>(n_keep_pre) * npw;
        for (int i = 0; i < npw; ++i) hdst[i] = hsrc[i] * inv_nrm;
      }
      ++n_keep_pre;
    }
  }
  if (n_keep_pre == 0) return 0;

  std::vector<Complex> S(static_cast<std::size_t>(n_keep_pre) * n_keep_pre);
  zgemm_host('C', 'N', n_keep_pre, n_keep_pre, npw, Complex(1.0, 0.0), V, npw, V, npw,
             Complex(0.0, 0.0), S.data(), n_keep_pre);

  std::vector<double> s_vals(static_cast<std::size_t>(n_keep_pre));
  std::vector<Complex> Z(static_cast<std::size_t>(n_keep_pre) * n_keep_pre);
  zheev_host(n_keep_pre, S.data(), s_vals.data(), Z.data());

  int n_out = 0;
  std::vector<Complex> T(static_cast<std::size_t>(n_keep_pre) * n_keep_pre, Complex(0.0, 0.0));
  for (int j = 0; j < n_keep_pre; ++j) {
    if (s_vals[j] > 1e-11) {
      const double inv_s = 1.0 / std::sqrt(s_vals[j]);
      for (int i = 0; i < n_keep_pre; ++i) {
        T[i + static_cast<std::size_t>(n_out) * n_keep_pre] =
            Z[i + static_cast<std::size_t>(j) * n_keep_pre] * inv_s;
      }
      ++n_out;
    }
  }
  if (n_out == 0) return 0;

  std::vector<Complex> V_new(static_cast<std::size_t>(npw) * n_out);
  zgemm_host('N', 'N', npw, n_out, n_keep_pre, Complex(1.0, 0.0), V, npw, T.data(), n_keep_pre,
             Complex(0.0, 0.0), V_new.data(), npw);
  std::copy(V_new.begin(), V_new.end(), V);

  if (HV_sync != nullptr) {
    std::vector<Complex> HV_new(static_cast<std::size_t>(npw) * n_out);
    zgemm_host('N', 'N', npw, n_out, n_keep_pre, Complex(1.0, 0.0), HV_sync, npw, T.data(),
               n_keep_pre, Complex(0.0, 0.0), HV_new.data(), npw);
    std::copy(HV_new.begin(), HV_new.end(), HV_sync);
  }
  return n_out;
}

}  // namespace

namespace detail {

void pw_zheev_host(int n, const Complex* a_col_major, double* w_out, Complex* u_col_major) {
  zheev_host(n, a_col_major, w_out, u_col_major);
}

void pw_cholesky_trsm_ace_host(int m,
                               int n,
                               const Complex* b_in,
                               const Complex* w_in,
                               Complex* xi_out) {
  cholesky_and_trsm_ace_host(m, n, b_in, w_in, xi_out);
}

}  // namespace detail

void NonLocalProjector::apply_add_host(const Complex* psi, Complex* hpsi, int nband) const {
  if (nproj_ == 0 || nband <= 0) return;
  std::vector<Complex> C(static_cast<std::size_t>(nproj_) * nband);
  std::vector<Complex> D(static_cast<std::size_t>(nproj_) * nband);
  zgemm_host('C', 'N', nproj_, nband, npw_, Complex(1.0, 0.0), P_.data(), npw_, psi, npw_,
             Complex(0.0, 0.0), C.data(), nproj_);
  zgemm_host('N', 'N', nproj_, nband, nproj_, Complex(1.0, 0.0), H_.data(), nproj_, C.data(),
             nproj_, Complex(0.0, 0.0), D.data(), nproj_);
  zgemm_host('N', 'N', npw_, nband, nproj_, Complex(1.0, 0.0), P_.data(), npw_, D.data(), nproj_,
             Complex(1.0, 0.0), hpsi, npw_);
}

// ============================================================================
// Host PW Hamiltonian Action, Davidson & LOBPCG Eigensolvers, XC & SCF Engine
// ============================================================================

namespace {

using HpsiCallbackHost = std::function<void(const Complex* psi, Complex* hpsi, int nband)>;

void apply_pw_hamiltonian_local_nl_host(const PwBasis& basis,
                                        int ik,
                                        const double* v_eff_r,
                                        const NonLocalProjector* nl_proj,
                                        const Complex* ace_xi,
                                        int ace_nocc,
                                        double alpha_x,
                                        const Complex* psi,
                                        Complex* hpsi,
                                        int nband) {
  const auto& kp = basis.kpoint(ik);
  const int npw = kp.npw;
  const int ngrid = basis.ngrid();
  const double inv_ngrid = 1.0 / static_cast<double>(ngrid);

  // 1. Diagonal kinetic energy (0.5 * |k + G|^2)
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int b = 0; b < nband; ++b) {
    const Complex* src = psi + static_cast<std::size_t>(b) * npw;
    Complex* dst = hpsi + static_cast<std::size_t>(b) * npw;
    for (int ig = 0; ig < npw; ++ig) {
      dst[ig] = kp.kin[static_cast<std::size_t>(ig)] * src[ig];
    }
  }

  // 2. Local effective potential via chunked batched 3D FFTs
  const int batch_chunk = std::min(nband, 32);
  std::vector<Complex> fft_buf(static_cast<std::size_t>(batch_chunk) * ngrid);
  for (int b0 = 0; b0 < nband; b0 += batch_chunk) {
    const int cur_b = std::min(batch_chunk, nband - b0);
    std::fill(fft_buf.begin(),
              fft_buf.begin() + static_cast<std::size_t>(cur_b) * ngrid,
              Complex(0.0, 0.0));
    for (int b = 0; b < cur_b; ++b) {
      const Complex* src = psi + static_cast<std::size_t>(b0 + b) * npw;
      Complex* box = fft_buf.data() + static_cast<std::size_t>(b) * ngrid;
      for (int ig = 0; ig < npw; ++ig) {
        box[kp.fft_indices[static_cast<std::size_t>(ig)]] = src[ig];
      }
    }

    fft3d_batch_host(basis.fft_grid(), cur_b, fft_buf.data(), fft_buf.data(), +1);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < cur_b; ++b) {
      Complex* box = fft_buf.data() + static_cast<std::size_t>(b) * ngrid;
      for (int ir = 0; ir < ngrid; ++ir) {
        box[ir] *= (v_eff_r[ir] * inv_ngrid);
      }
    }

    fft3d_batch_host(basis.fft_grid(), cur_b, fft_buf.data(), fft_buf.data(), -1);

    for (int b = 0; b < cur_b; ++b) {
      const Complex* box = fft_buf.data() + static_cast<std::size_t>(b) * ngrid;
      Complex* dst = hpsi + static_cast<std::size_t>(b0 + b) * npw;
      for (int ig = 0; ig < npw; ++ig) {
        dst[ig] += box[kp.fft_indices[static_cast<std::size_t>(ig)]];
      }
    }
  }

  // 3. Separable GTH Non-Local Projectors via two ZGEMMs
  if (nl_proj != nullptr && nl_proj->nproj() > 0) {
    nl_proj->apply_add_host(psi, hpsi, nband);
  }

  // 4. Low-rank ACE Hybrid Exact Exchange via two ZGEMMs (ZERO 3D FFTs!)
  if (ace_xi != nullptr && ace_nocc > 0 && std::abs(alpha_x) > 0.0) {
    std::vector<Complex> overlap(static_cast<std::size_t>(ace_nocc) * nband);
    zgemm_host('C', 'N', ace_nocc, nband, npw, Complex(1.0, 0.0), ace_xi, npw, psi, npw,
               Complex(0.0, 0.0), overlap.data(), ace_nocc);
    zgemm_host('N', 'N', npw, nband, ace_nocc, Complex(-alpha_x, 0.0), ace_xi, npw, overlap.data(),
               ace_nocc, Complex(1.0, 0.0), hpsi, npw);
  }
}

// Deterministic crystal-symmetry-friendly plane-wave initial guess (lowest kinetic plane waves + subspace Ritz)
void build_initial_guess_host(const KPointBasis& kp,
                              int n_solve,
                              const HpsiCallbackHost& apply_h,
                              std::vector<Complex>& psi_out,
                              std::vector<double>& evals_out,
                              int& hpsi_calls) {
  const int npw = kp.npw;
  const int n_init = std::min(npw, std::max(n_solve * 3, std::min(npw, 64)));

  std::vector<int> order(static_cast<std::size_t>(npw));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    if (std::abs(kp.kin[a] - kp.kin[b]) > 1e-12) return kp.kin[a] < kp.kin[b];
    return kp.fft_indices[a] < kp.fft_indices[b];
  });

  std::vector<Complex> V0(static_cast<std::size_t>(npw) * n_init, Complex(0.0, 0.0));
  for (int j = 0; j < n_init; ++j) {
    V0[order[static_cast<std::size_t>(j)] + static_cast<std::size_t>(j) * npw] = Complex(1.0, 0.0);
  }

  std::vector<Complex> HV0(static_cast<std::size_t>(npw) * n_init);
  apply_h(V0.data(), HV0.data(), n_init);
  hpsi_calls += n_init;

  std::vector<Complex> H_sub(static_cast<std::size_t>(n_init) * n_init);
  zgemm_host('C', 'N', n_init, n_init, npw, Complex(1.0, 0.0), V0.data(), npw, HV0.data(), npw,
             Complex(0.0, 0.0), H_sub.data(), n_init);

  std::vector<double> w_sub(static_cast<std::size_t>(n_init));
  std::vector<Complex> U_sub(static_cast<std::size_t>(n_init) * n_init);
  zheev_host(n_init, H_sub.data(), w_sub.data(), U_sub.data());

  psi_out.resize(static_cast<std::size_t>(npw) * n_solve);
  zgemm_host('N', 'N', npw, n_solve, n_init, Complex(1.0, 0.0), V0.data(), npw, U_sub.data(),
             n_init, Complex(0.0, 0.0), psi_out.data(), npw);
  evals_out.assign(w_sub.begin(), w_sub.begin() + n_solve);
}

// Block-Davidson & Hetmaniuk-Lehoucq LOBPCG Iterative Eigensolvers on Host
PwEigenResult run_eigensolver_host(const KPointBasis& kp,
                                   int nband_target,
                                   int n_solve,
                                   PwSolverType solver,
                                   double residual_tol,
                                   int max_iter,
                                   const HpsiCallbackHost& apply_h,
                                   std::vector<Complex>& psi_inout) {
  const int npw = kp.npw;
  n_solve = std::min(npw, std::max(nband_target, n_solve));
  PwEigenResult res;

  // Exact dense diagonalization shortcut when subspace equals full Hilbert space
  if (npw <= n_solve) {
    std::vector<Complex> I_mat(static_cast<std::size_t>(npw) * npw, Complex(0.0, 0.0));
    for (int i = 0; i < npw; ++i) I_mat[i + static_cast<std::size_t>(i) * npw] = Complex(1.0, 0.0);
    std::vector<Complex> H_mat(static_cast<std::size_t>(npw) * npw);
    apply_h(I_mat.data(), H_mat.data(), npw);
    res.hpsi_calls += npw;
    res.iterations = 1;
    std::vector<double> w_all(static_cast<std::size_t>(npw));
    std::vector<Complex> U_all(static_cast<std::size_t>(npw) * npw);
    zheev_host(npw, H_mat.data(), w_all.data(), U_all.data());
    psi_inout = U_all;
    res.eigenvalues.assign(w_all.begin(), w_all.begin() + nband_target);
    res.eigenvectors.assign(U_all.begin(), U_all.begin() + static_cast<std::size_t>(npw) * nband_target);
    return res;
  }

  if (static_cast<int>(psi_inout.size()) != npw * n_solve) {
    std::vector<double> init_w;
    build_initial_guess_host(kp, n_solve, apply_h, psi_inout, init_w, res.hpsi_calls);
  } else {
    const int kept = svqb_orthonormalize_host(npw, psi_inout.data(), n_solve, nullptr, 0);
    if (kept < n_solve) {
      std::vector<double> init_w;
      build_initial_guess_host(kp, n_solve, apply_h, psi_inout, init_w, res.hpsi_calls);
    }
  }

  if (solver == PwSolverType::Davidson) {
    const int max_sub = std::min(npw, std::max(2 * n_solve + 4, 4 * n_solve));
    std::vector<Complex> V_sub(static_cast<std::size_t>(npw) * max_sub);
    std::vector<Complex> HV_sub(static_cast<std::size_t>(npw) * max_sub);
    std::copy(psi_inout.begin(), psi_inout.end(), V_sub.begin());
    apply_h(V_sub.data(), HV_sub.data(), n_solve);
    res.hpsi_calls += n_solve;
    int nsub = n_solve;

    std::vector<Complex> Psi(static_cast<std::size_t>(npw) * n_solve);
    std::vector<Complex> HPsi(static_cast<std::size_t>(npw) * n_solve);
    std::vector<double> theta(static_cast<std::size_t>(n_solve), 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
      res.iterations = iter + 1;
      std::vector<Complex> H_proj(static_cast<std::size_t>(nsub) * nsub);
      zgemm_host('C', 'N', nsub, nsub, npw, Complex(1.0, 0.0), V_sub.data(), npw, HV_sub.data(),
                 npw, Complex(0.0, 0.0), H_proj.data(), nsub);

      std::vector<double> w_sub(static_cast<std::size_t>(nsub));
      std::vector<Complex> U_sub(static_cast<std::size_t>(nsub) * nsub);
      zheev_host(nsub, H_proj.data(), w_sub.data(), U_sub.data());
      std::copy(w_sub.begin(), w_sub.begin() + n_solve, theta.begin());

      zgemm_host('N', 'N', npw, n_solve, nsub, Complex(1.0, 0.0), V_sub.data(), npw, U_sub.data(),
                 nsub, Complex(0.0, 0.0), Psi.data(), npw);
      zgemm_host('N', 'N', npw, n_solve, nsub, Complex(1.0, 0.0), HV_sub.data(), npw, U_sub.data(),
                 nsub, Complex(0.0, 0.0), HPsi.data(), npw);

      // Compute residuals and Teter-Payne-Allan preconditioned search vectors
      std::vector<Complex> W_cand(static_cast<std::size_t>(npw) * n_solve);
      int n_unconv = 0;
      double max_res = 0.0;
      for (int b = 0; b < n_solve; ++b) {
        const Complex* p_col = Psi.data() + static_cast<std::size_t>(b) * npw;
        const Complex* hp_col = HPsi.data() + static_cast<std::size_t>(b) * npw;
        double res2 = 0.0;
        double ekin_b = 0.0;
        for (int ig = 0; ig < npw; ++ig) {
          const Complex r = hp_col[ig] - theta[b] * p_col[ig];
          res2 += std::norm(r);
          ekin_b += kp.kin[ig] * std::norm(p_col[ig]);
        }
        const double res_norm = std::sqrt(res2);
        if (b < nband_target) max_res = std::max(max_res, res_norm);

        const double band_tol = (b < nband_target) ? residual_tol : (10.0 * residual_tol);
        if (res_norm > band_tol) {
          Complex* w_col = W_cand.data() + static_cast<std::size_t>(n_unconv) * npw;
          for (int ig = 0; ig < npw; ++ig) {
            const Complex r = hp_col[ig] - theta[b] * p_col[ig];
            w_col[ig] = detail::teter_preconditioner(kp.kin[ig], ekin_b) * r;
          }
          ++n_unconv;
        }
      }
      res.max_residual = max_res;
      if (max_res < residual_tol || n_unconv == 0 || iter + 1 == max_iter) break;

      // Restart subspace if full
      if (nsub + n_unconv > max_sub) {
        std::copy(Psi.begin(), Psi.end(), V_sub.begin());
        std::copy(HPsi.begin(), HPsi.end(), HV_sub.begin());
        nsub = n_solve;
      }

      const int n_add = svqb_orthonormalize_host(npw, W_cand.data(), n_unconv, V_sub.data(), nsub);
      if (n_add == 0) break;

      Complex* V_next = V_sub.data() + static_cast<std::size_t>(nsub) * npw;
      Complex* HV_next = HV_sub.data() + static_cast<std::size_t>(nsub) * npw;
      std::copy(W_cand.begin(), W_cand.begin() + static_cast<std::size_t>(npw) * n_add, V_next);
      apply_h(V_next, HV_next, n_add);
      res.hpsi_calls += n_add;
      nsub += n_add;
    }

    psi_inout = Psi;
    res.eigenvalues.assign(theta.begin(), theta.begin() + nband_target);
    res.eigenvectors.assign(Psi.begin(), Psi.begin() + static_cast<std::size_t>(npw) * nband_target);
    return res;
  }

  // Hetmaniuk-Lehoucq Orthonormalized LOBPCG
  std::vector<Complex> X = psi_inout;
  std::vector<Complex> HX(static_cast<std::size_t>(npw) * n_solve);
  apply_h(X.data(), HX.data(), n_solve);
  res.hpsi_calls += n_solve;

  // Initial Rayleigh-Ritz on X
  {
    std::vector<Complex> Hx(static_cast<std::size_t>(n_solve) * n_solve);
    zgemm_host('C', 'N', n_solve, n_solve, npw, Complex(1.0, 0.0), X.data(), npw, HX.data(), npw,
               Complex(0.0, 0.0), Hx.data(), n_solve);
    std::vector<double> wx(static_cast<std::size_t>(n_solve));
    std::vector<Complex> Ux(static_cast<std::size_t>(n_solve) * n_solve);
    zheev_host(n_solve, Hx.data(), wx.data(), Ux.data());
    std::vector<Complex> X_rot(static_cast<std::size_t>(npw) * n_solve);
    std::vector<Complex> HX_rot(static_cast<std::size_t>(npw) * n_solve);
    zgemm_host('N', 'N', npw, n_solve, n_solve, Complex(1.0, 0.0), X.data(), npw, Ux.data(),
               n_solve, Complex(0.0, 0.0), X_rot.data(), npw);
    zgemm_host('N', 'N', npw, n_solve, n_solve, Complex(1.0, 0.0), HX.data(), npw, Ux.data(),
               n_solve, Complex(0.0, 0.0), HX_rot.data(), npw);
    X.swap(X_rot);
    HX.swap(HX_rot);
    res.eigenvalues.assign(wx.begin(), wx.begin() + nband_target);
  }

  std::vector<double> lambda(static_cast<std::size_t>(n_solve), 0.0);
  for (int b = 0; b < n_solve; ++b) {
    Complex rq(0.0, 0.0);
    for (int ig = 0; ig < npw; ++ig) {
      rq += std::conj(X[ig + static_cast<std::size_t>(b) * npw]) *
            HX[ig + static_cast<std::size_t>(b) * npw];
    }
    lambda[b] = rq.real();
  }

  std::vector<Complex> P;
  std::vector<Complex> HP;
  int n_p = 0;

  for (int iter = 0; iter < max_iter; ++iter) {
    res.iterations = iter + 1;
    std::vector<Complex> W(static_cast<std::size_t>(npw) * n_solve);
    int n_w = 0;
    double max_res = 0.0;
    for (int b = 0; b < n_solve; ++b) {
      const Complex* x_col = X.data() + static_cast<std::size_t>(b) * npw;
      const Complex* hx_col = HX.data() + static_cast<std::size_t>(b) * npw;
      double res2 = 0.0;
      double ekin_b = 0.0;
      for (int ig = 0; ig < npw; ++ig) {
        const Complex r = hx_col[ig] - lambda[b] * x_col[ig];
        res2 += std::norm(r);
        ekin_b += kp.kin[ig] * std::norm(x_col[ig]);
      }
      const double rn = std::sqrt(res2);
      if (b < nband_target) max_res = std::max(max_res, rn);
      const double band_tol = (b < nband_target) ? residual_tol : (10.0 * residual_tol);
      if (rn > band_tol) {
        Complex* w_col = W.data() + static_cast<std::size_t>(n_w) * npw;
        for (int ig = 0; ig < npw; ++ig) {
          const Complex r = hx_col[ig] - lambda[b] * x_col[ig];
          w_col[ig] = detail::teter_preconditioner(kp.kin[ig], ekin_b) * r;
        }
        ++n_w;
      }
    }
    res.max_residual = max_res;
    if (max_res < residual_tol || n_w == 0) break;

    n_w = svqb_orthonormalize_host(npw, W.data(), n_w, X.data(), n_solve);
    if (n_w == 0) break;

    std::vector<Complex> HW(static_cast<std::size_t>(npw) * n_w);
    apply_h(W.data(), HW.data(), n_w);
    res.hpsi_calls += n_w;

    // Assemble V = [X, W, P_orth] and HV = [HX, HW, HP_orth]
    const int n_xw = n_solve + n_w;
    std::vector<Complex> V(static_cast<std::size_t>(npw) * (n_xw + n_p));
    std::vector<Complex> HV(static_cast<std::size_t>(npw) * (n_xw + n_p));
    std::copy(X.begin(), X.end(), V.begin());
    std::copy(HX.begin(), HX.end(), HV.begin());
    std::copy(W.begin(), W.begin() + static_cast<std::size_t>(npw) * n_w,
              V.begin() + static_cast<std::size_t>(npw) * n_solve);
    std::copy(HW.begin(), HW.end(), HV.begin() + static_cast<std::size_t>(npw) * n_solve);

    int n_p_act = 0;
    if (n_p > 0 && n_xw < npw) {
      n_p_act = std::min(n_p, npw - n_xw);
      std::copy(P.begin(), P.begin() + static_cast<std::size_t>(npw) * n_p_act,
                V.begin() + static_cast<std::size_t>(npw) * n_xw);
      std::copy(HP.begin(), HP.begin() + static_cast<std::size_t>(npw) * n_p_act,
                HV.begin() + static_cast<std::size_t>(npw) * n_xw);
      n_p_act = svqb_orthonormalize_host(
          npw,
          V.data() + static_cast<std::size_t>(npw) * n_xw,
          n_p_act,
          V.data(),
          n_xw,
          HV.data() + static_cast<std::size_t>(npw) * n_xw,
          HV.data());
    }

    const int nsub = n_xw + n_p_act;
    std::vector<Complex> H_sub(static_cast<std::size_t>(nsub) * nsub);
    zgemm_host('C', 'N', nsub, nsub, npw, Complex(1.0, 0.0), V.data(), npw, HV.data(), npw,
               Complex(0.0, 0.0), H_sub.data(), nsub);

    std::vector<double> w_sub(static_cast<std::size_t>(nsub));
    std::vector<Complex> U_sub(static_cast<std::size_t>(nsub) * nsub);
    zheev_host(nsub, H_sub.data(), w_sub.data(), U_sub.data());
    std::copy(w_sub.begin(), w_sub.begin() + n_solve, lambda.begin());

    zgemm_host('N', 'N', npw, n_solve, nsub, Complex(1.0, 0.0), V.data(), npw, U_sub.data(), nsub,
               Complex(0.0, 0.0), X.data(), npw);
    zgemm_host('N', 'N', npw, n_solve, nsub, Complex(1.0, 0.0), HV.data(), npw, U_sub.data(), nsub,
               Complex(0.0, 0.0), HX.data(), npw);

    // Conjugate search direction P = V[:, n_solve:nsub] * U_sub[n_solve:nsub, 0:n_solve]
    const int n_wp = nsub - n_solve;
    P.resize(static_cast<std::size_t>(npw) * n_solve);
    HP.resize(static_cast<std::size_t>(npw) * n_solve);
    zgemm_host('N', 'N', npw, n_solve, n_wp, Complex(1.0, 0.0),
               V.data() + static_cast<std::size_t>(npw) * n_solve, npw,
               U_sub.data() + n_solve, nsub,
               Complex(0.0, 0.0), P.data(), npw);
    zgemm_host('N', 'N', npw, n_solve, n_wp, Complex(1.0, 0.0),
               HV.data() + static_cast<std::size_t>(npw) * n_solve, npw,
               U_sub.data() + n_solve, nsub,
               Complex(0.0, 0.0), HP.data(), npw);
    n_p = n_solve;
  }

  psi_inout = X;
  res.eigenvalues.assign(lambda.begin(), lambda.begin() + nband_target);
  res.eigenvectors.assign(X.begin(), X.begin() + static_cast<std::size_t>(npw) * nband_target);
  return res;
}

}  // namespace

PwEigenResult solve_pw_eigenproblem(const PwBasis& basis,
                                    int ik,
                                    std::span<const double> vloc_r,
                                    int nband,
                                    PwSolverType solver,
                                    Backend backend,
                                    double residual_tol,
                                    int max_iter,
                                    bool include_nonlocal_pp) {
  if (static_cast<int>(vloc_r.size()) != basis.ngrid()) {
    throw std::invalid_argument("solve_pw_eigenproblem: vloc_r size must equal basis.ngrid()");
  }
  if (nband <= 0 || nband > basis.kpoint(ik).npw) {
    throw std::invalid_argument("solve_pw_eigenproblem: invalid nband");
  }
  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::pw_cuda_device_count() == 0) {
      throw std::runtime_error("solve_pw_eigenproblem: no CUDA device available");
    }
    return detail::pw_cuda_solve_eigenproblem(
        basis, ik, vloc_r, nband, solver, residual_tol, max_iter, include_nonlocal_pp);
#else
    throw std::runtime_error("solve_pw_eigenproblem: gpu4dft was built without CUDA");
#endif
  }
  if (backend != Backend::Host) {
    throw std::runtime_error("solve_pw_eigenproblem: unsupported backend");
  }

  NonLocalProjector nl;
  if (include_nonlocal_pp) {
    nl = NonLocalProjector(basis.cell(), basis.kpoint(ik));
  }
  const int n_solve = std::min(basis.kpoint(ik).npw, std::max(nband + 4, nband * 2));
  std::vector<Complex> psi;
  auto apply_h = [&](const Complex* x, Complex* hx, int nb) {
    apply_pw_hamiltonian_local_nl_host(basis, ik, vloc_r.data(),
                                       include_nonlocal_pp ? &nl : nullptr,
                                       nullptr, 0, 0.0, x, hx, nb);
  };
  return run_eigensolver_host(basis.kpoint(ik), nband, n_solve, solver, residual_tol, max_iter,
                              apply_h, psi);
}

// ============================================================================
// Host Plane-Wave Kohn-Sham + Hybrid ACE SCF Engine
// ============================================================================

namespace {

// Evaluates Hartree potential V_H(r), Hartree energy E_H, and symbolic LDA/GGA V_xc(r) + E_xc
// on the uniform FFT grid using exact spectral derivatives (zero finite-difference error).
void eval_hartree_and_xc_host(const PwBasis& basis,
                              xc_gen::XcFunctionalId xc_id,
                              double rho_cutoff,
                              const std::vector<double>& rho_r,
                              std::vector<Complex>& rho_g_out,
                              std::vector<double>& vh_r_out,
                              std::vector<double>& vxc_r_out,
                              double& eh_out,
                              double& exc_out) {
  const int ngrid = basis.ngrid();
  const double vol = basis.cell().volume();
  const double dvol = vol / static_cast<double>(ngrid);
  const double inv_ngrid = 1.0 / static_cast<double>(ngrid);

  std::vector<Complex> buf_c(static_cast<std::size_t>(ngrid));
  for (int ir = 0; ir < ngrid; ++ir) buf_c[ir] = Complex(rho_r[ir], 0.0);
  rho_g_out.resize(static_cast<std::size_t>(ngrid));
  fft3d_batch_host(basis.fft_grid(), 1, buf_c.data(), rho_g_out.data(), -1);
  for (int ig = 0; ig < ngrid; ++ig) rho_g_out[ig] *= inv_ngrid;

  // Hartree potential V_H(G) = 4*pi * rho(G) / |G|^2 (G != 0)
  const auto& g2 = basis.g2();
  double eh = 0.0;
  for (int ig = 0; ig < ngrid; ++ig) {
    if (g2[ig] > 1e-18) {
      const double fac = kFourPi / g2[ig];
      buf_c[ig] = fac * rho_g_out[ig];
      eh += 0.5 * vol * fac * std::norm(rho_g_out[ig]);
    } else {
      buf_c[ig] = Complex(0.0, 0.0);
    }
  }
  eh_out = eh;
  fft3d_batch_host(basis.fft_grid(), 1, buf_c.data(), buf_c.data(), +1);
  vh_r_out.resize(static_cast<std::size_t>(ngrid));
  for (int ir = 0; ir < ngrid; ++ir) vh_r_out[ir] = buf_c[ir].real();

  vxc_r_out.assign(static_cast<std::size_t>(ngrid), 0.0);
  exc_out = 0.0;
  if (xc_id == xc_gen::XcFunctionalId::k_hf) return;

  const auto fam = xc_gen::xc_functional_family(xc_id);
  const bool is_gga = (fam != xc_gen::XcFamily::LDA);
  const auto& gv = basis.g_vecs_xyz();

  std::vector<double> gx_rho, gy_rho, gz_rho, sigma_r;
  std::vector<Complex> grad_buf;
  if (is_gga) {
    grad_buf.resize(static_cast<std::size_t>(3) * ngrid);
    for (int d = 0; d < 3; ++d) {
      Complex* dst = grad_buf.data() + static_cast<std::size_t>(d) * ngrid;
      for (int ig = 0; ig < ngrid; ++ig) {
        const double g_comp = gv[3 * static_cast<std::size_t>(ig) + d];
        dst[ig] = Complex(0.0, g_comp) * rho_g_out[ig];
      }
    }
    fft3d_batch_host(basis.fft_grid(), 3, grad_buf.data(), grad_buf.data(), +1);
    gx_rho.resize(static_cast<std::size_t>(ngrid));
    gy_rho.resize(static_cast<std::size_t>(ngrid));
    gz_rho.resize(static_cast<std::size_t>(ngrid));
    sigma_r.resize(static_cast<std::size_t>(ngrid));
    for (int ir = 0; ir < ngrid; ++ir) {
      const double rx = grad_buf[ir].real();
      const double ry = grad_buf[ngrid + ir].real();
      const double rz = grad_buf[2 * ngrid + ir].real();
      gx_rho[ir] = rx;
      gy_rho[ir] = ry;
      gz_rho[ir] = rz;
      sigma_r[ir] = rx * rx + ry * ry + rz * rz;
    }
  }

  std::vector<double> vsigma_r;
  if (is_gga) vsigma_r.assign(static_cast<std::size_t>(ngrid), 0.0);

  double exc_sum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for reduction(+ : exc_sum) schedule(static)
#endif
  for (int ir = 0; ir < ngrid; ++ir) {
    const double r = rho_r[ir];
    const double s = is_gga ? sigma_r[ir] : 0.0;
    xc_gen::XcPointOut pt{};
    xc_gen::eval_xc_point(xc_id, r, s, 0.0, 1, pt, rho_cutoff);
    if (r >= rho_cutoff) {
      exc_sum += r * pt.zk;
    }
    vxc_r_out[ir] = pt.vrho;
    if (is_gga) vsigma_r[ir] = pt.vsigma;
  }
  exc_out = exc_sum * dvol;

  if (is_gga) {
    // Spectral divergence -div(2 * vsigma * grad(rho))
    for (int ir = 0; ir < ngrid; ++ir) {
      const double two_vs = 2.0 * vsigma_r[ir];
      grad_buf[ir] = Complex(two_vs * gx_rho[ir], 0.0);
      grad_buf[ngrid + ir] = Complex(two_vs * gy_rho[ir], 0.0);
      grad_buf[2 * ngrid + ir] = Complex(two_vs * gz_rho[ir], 0.0);
    }
    fft3d_batch_host(basis.fft_grid(), 3, grad_buf.data(), grad_buf.data(), -1);
    for (int ig = 0; ig < ngrid; ++ig) {
      const double gx = gv[3 * static_cast<std::size_t>(ig) + 0];
      const double gy = gv[3 * static_cast<std::size_t>(ig) + 1];
      const double gz = gv[3 * static_cast<std::size_t>(ig) + 2];
      const Complex hx = grad_buf[ig] * inv_ngrid;
      const Complex hy = grad_buf[ngrid + ig] * inv_ngrid;
      const Complex hz = grad_buf[2 * ngrid + ig] * inv_ngrid;
      // -i * (G . h(G))
      buf_c[ig] = Complex(0.0, -1.0) * (gx * hx + gy * hy + gz * hz);
    }
    fft3d_batch_host(basis.fft_grid(), 1, buf_c.data(), buf_c.data(), +1);
    for (int ir = 0; ir < ngrid; ++ir) {
      vxc_r_out[ir] += buf_c[ir].real();
    }
  }
}

// Accumulates rho(r) = sum_{k, n < n_occ} w_k * 2.0 * |psi_{n,k}(r)|^2
void compute_density_host(const PwBasis& basis,
                          int n_occ,
                          const std::vector<std::vector<Complex>>& psi_k,
                          std::vector<double>& rho_r_out) {
  const int ngrid = basis.ngrid();
  const double inv_vol = 1.0 / basis.cell().volume();
  rho_r_out.assign(static_cast<std::size_t>(ngrid), 0.0);
  if (n_occ <= 0) return;

  std::vector<Complex> fft_buf(static_cast<std::size_t>(n_occ) * ngrid);
  for (int ik = 0; ik < basis.nkpts(); ++ik) {
    const auto& kp = basis.kpoint(ik);
    const double wk_occ = 2.0 * kp.weight * inv_vol;
    std::fill(fft_buf.begin(), fft_buf.end(), Complex(0.0, 0.0));
    for (int b = 0; b < n_occ; ++b) {
      const Complex* src = psi_k[ik].data() + static_cast<std::size_t>(b) * kp.npw;
      Complex* box = fft_buf.data() + static_cast<std::size_t>(b) * ngrid;
      for (int ig = 0; ig < kp.npw; ++ig) {
        box[kp.fft_indices[ig]] = src[ig];
      }
    }
    fft3d_batch_host(basis.fft_grid(), n_occ, fft_buf.data(), fft_buf.data(), +1);
    for (int b = 0; b < n_occ; ++b) {
      const Complex* box = fft_buf.data() + static_cast<std::size_t>(b) * ngrid;
      for (int ir = 0; ir < ngrid; ++ir) {
        rho_r_out[ir] += wk_occ * std::norm(box[ir]);
      }
    }
  }
}

// Initial superposition of smooth atomic valence Gaussians (exact total electron count N_elec)
std::vector<double> initial_atomic_gaussian_density(const PwBasis& basis, int n_elec) {
  const int ngrid = basis.ngrid();
  const double vol = basis.cell().volume();
  const double z_tot = basis.cell().total_valence_charge();
  const double scale_q = (z_tot > 0.0) ? (static_cast<double>(n_elec) / z_tot) : 0.0;

  std::vector<Complex> rho_g(static_cast<std::size_t>(ngrid), Complex(0.0, 0.0));
  const auto& gv = basis.g_vecs_xyz();
  const auto& g2 = basis.g2();

  if (basis.cell().natom() == 0 || z_tot <= 0.0) {
    rho_g[0] = Complex(static_cast<double>(n_elec) / vol, 0.0);
  } else {
    for (const auto& atm : basis.cell().atoms()) {
      const double q_atm = scale_q * atm.pseudo.z_ion;
      const double sigma = std::max(0.75, 1.8 * atm.pseudo.r_loc);
      const double half_s2 = 0.5 * sigma * sigma;
      for (int ig = 0; ig < ngrid; ++ig) {
        const double phase = -(gv[3 * static_cast<std::size_t>(ig) + 0] * atm.pos[0] +
                               gv[3 * static_cast<std::size_t>(ig) + 1] * atm.pos[1] +
                               gv[3 * static_cast<std::size_t>(ig) + 2] * atm.pos[2]);
        const double env = (q_atm / vol) * std::exp(-half_s2 * g2[ig]);
        rho_g[ig] += env * Complex(std::cos(phase), std::sin(phase));
      }
    }
    rho_g[0] = Complex(static_cast<double>(n_elec) / vol, 0.0);
  }

  std::vector<Complex> rho_r_c(static_cast<std::size_t>(ngrid));
  fft3d_batch_host(basis.fft_grid(), 1, rho_g.data(), rho_r_c.data(), +1);
  std::vector<double> rho_r(static_cast<std::size_t>(ngrid));
  for (int ir = 0; ir < ngrid; ++ir) {
    rho_r[ir] = std::max(1e-14, rho_r_c[ir].real());
  }
  return rho_r;
}

// Kerker-preconditioned Pulay Density DIIS in G-space
class KerkerPulayDiis {
 public:
  KerkerPulayDiis(int ngrid, const std::vector<double>& g2, double vol,
                  double beta, double q0, int max_vec)
      : ngrid_(ngrid), g2_(g2), vol_(vol), beta_(beta), q0_sq_(q0 * q0), max_vec_(std::max(1, max_vec)) {}

  void clear() {
    rho_in_hist_.clear();
    res_hist_.clear();
  }

  // Returns G-space L2 RMS residual and writes next mixed real-space density `rho_next_r`
  double step(const std::array<int, 3>& fft_grid,
              double n_elec,
              const std::vector<Complex>& rho_in_g,
              const std::vector<Complex>& rho_out_g,
              std::vector<double>& rho_next_r) {
    std::vector<Complex> res_g(static_cast<std::size_t>(ngrid_));
    double rms2 = 0.0;
    for (int ig = 0; ig < ngrid_; ++ig) {
      res_g[ig] = rho_out_g[ig] - rho_in_g[ig];
      rms2 += std::norm(res_g[ig]);
    }
    res_g[0] = Complex(0.0, 0.0);
    const double rms = std::sqrt(rms2 * vol_);

    if (static_cast<int>(rho_in_hist_.size()) >= max_vec_) {
      rho_in_hist_.erase(rho_in_hist_.begin());
      res_hist_.erase(res_hist_.begin());
    }
    rho_in_hist_.push_back(rho_in_g);
    res_hist_.push_back(std::move(res_g));

    const int m = static_cast<int>(rho_in_hist_.size());
    std::vector<double> weights(static_cast<std::size_t>(m), 0.0);
    weights[m - 1] = 1.0;

    if (m >= 2) {
      // Build (m + 1) x (m + 1) Pulay matrix using Kerker-weighted inner product
      const int dim = m + 1;
      std::vector<double> A(static_cast<std::size_t>(dim) * dim, 0.0);
      std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
      double max_diag = 0.0;
      for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
          double dot = 0.0;
          for (int ig = 1; ig < ngrid_; ++ig) {
            const double w_kerker = (g2_[ig] + q0_sq_) / std::max(g2_[ig], 1e-12);
            dot += w_kerker * (res_hist_[i][ig].real() * res_hist_[j][ig].real() +
                               res_hist_[i][ig].imag() * res_hist_[j][ig].imag());
          }
          dot *= vol_;
          A[i * dim + j] = dot;
          A[j * dim + i] = dot;
          if (i == j) max_diag = std::max(max_diag, dot);
        }
        A[i * dim + m] = 1.0;
        A[m * dim + i] = 1.0;
      }
      rhs[m] = 1.0;
      const double reg = std::max(1e-28, 1e-13 * max_diag);
      for (int i = 0; i < m; ++i) A[i * dim + i] += reg;

      // Solve small linear system via Gaussian elimination with partial pivoting
      bool ok = true;
      for (int k = 0; k < dim; ++k) {
        int piv = k;
        double max_v = std::abs(A[k * dim + k]);
        for (int i = k + 1; i < dim; ++i) {
          if (std::abs(A[i * dim + k]) > max_v) {
            max_v = std::abs(A[i * dim + k]);
            piv = i;
          }
        }
        if (max_v < 1e-30) {
          ok = false;
          break;
        }
        if (piv != k) {
          for (int j = k; j < dim; ++j) std::swap(A[k * dim + j], A[piv * dim + j]);
          std::swap(rhs[k], rhs[piv]);
        }
        for (int i = k + 1; i < dim; ++i) {
          const double f = A[i * dim + k] / A[k * dim + k];
          for (int j = k; j < dim; ++j) A[i * dim + j] -= f * A[k * dim + j];
          rhs[i] -= f * rhs[k];
        }
      }
      if (ok) {
        std::vector<double> sol(static_cast<std::size_t>(dim), 0.0);
        for (int i = dim - 1; i >= 0; --i) {
          double s = rhs[i];
          for (int j = i + 1; j < dim; ++j) s -= A[i * dim + j] * sol[j];
          sol[i] = s / A[i * dim + i];
        }
        double max_abs_w = 0.0;
        for (int i = 0; i < m; ++i) max_abs_w = std::max(max_abs_w, std::abs(sol[i]));
        if (std::isfinite(max_abs_w) && max_abs_w < 20.0) {
          for (int i = 0; i < m; ++i) weights[i] = sol[i];
        }
      }
    }

    std::vector<Complex> mixed_g(static_cast<std::size_t>(ngrid_), Complex(0.0, 0.0));
    mixed_g[0] = Complex(n_elec / vol_, 0.0);
    for (int ig = 1; ig < ngrid_; ++ig) {
      Complex r_opt(0.0, 0.0);
      Complex res_opt(0.0, 0.0);
      for (int a = 0; a < m; ++a) {
        r_opt += weights[a] * rho_in_hist_[a][ig];
        res_opt += weights[a] * res_hist_[a][ig];
      }
      const double kerker_damp = beta_ * (g2_[ig] / (g2_[ig] + q0_sq_));
      mixed_g[ig] = r_opt + kerker_damp * res_opt;
    }

    std::vector<Complex> mixed_r_c(static_cast<std::size_t>(ngrid_));
    fft3d_batch_host(fft_grid, 1, mixed_g.data(), mixed_r_c.data(), +1);
    rho_next_r.resize(static_cast<std::size_t>(ngrid_));
    for (int ir = 0; ir < ngrid_; ++ir) {
      rho_next_r[ir] = mixed_r_c[ir].real();
    }
    return rms;
  }

 private:
  int ngrid_ = 0;
  const std::vector<double>& g2_;
  double vol_ = 1.0;
  double beta_ = 0.6;
  double q0_sq_ = 0.64;
  int max_vec_ = 8;
  std::vector<std::vector<Complex>> rho_in_hist_;
  std::vector<std::vector<Complex>> res_hist_;
};

// Exact Exchange Action W_k = V_X[{Psi_occ,q}] * Psi_trial,k via 3D Poisson FFTs on Host.
// When `symmetric_occ_square == true` (Psi_trial == Psi_occ and nk == 1), exploits (i <= j) symmetry!
void apply_exact_exchange_poisson_host(const PwBasis& basis,
                                       int ik_target,
                                       int n_occ,
                                       const std::vector<std::vector<Complex>>& u_occ_r_all_k,
                                       const Complex* psi_trial,
                                       int n_trial,
                                       bool symmetric_occ_square,
                                       ExxDiv exxdiv,
                                       std::array<int, 3> kmesh,
                                       Complex* vx_psi_out,
                                       int& poisson_fft_counter) {
  const int ngrid = basis.ngrid();
  const double vol = basis.cell().volume();
  const double inv_sqrt_vol = 1.0 / std::sqrt(vol);
  const double inv_ngrid = 1.0 / static_cast<double>(ngrid);
  const auto& kp_t = basis.kpoint(ik_target);
  const int npw = kp_t.npw;

  // Transform trial orbitals u_trial(r) = (1/sqrt(vol)) * FFT_INV(psi_trial)
  std::vector<Complex> u_trial_r(static_cast<std::size_t>(n_trial) * ngrid, Complex(0.0, 0.0));
  for (int b = 0; b < n_trial; ++b) {
    const Complex* src = psi_trial + static_cast<std::size_t>(b) * npw;
    Complex* box = u_trial_r.data() + static_cast<std::size_t>(b) * ngrid;
    for (int ig = 0; ig < npw; ++ig) box[kp_t.fft_indices[ig]] = src[ig];
  }
  fft3d_batch_host(basis.fft_grid(), n_trial, u_trial_r.data(), u_trial_r.data(), +1);
  for (auto& val : u_trial_r) val *= inv_sqrt_vol;

  std::vector<Complex> w_r(static_cast<std::size_t>(n_trial) * ngrid, Complex(0.0, 0.0));

  for (int iq = 0; iq < basis.nkpts(); ++iq) {
    const auto& kp_q = basis.kpoint(iq);
    const double wq = kp_q.weight;
    const std::array<double, 3> q_shift{
        kp_t.k_cart[0] - kp_q.k_cart[0],
        kp_t.k_cart[1] - kp_q.k_cart[1],
        kp_t.k_cart[2] - kp_q.k_cart[2],
    };
    const auto coul_g = basis.compute_exx_coulomb_kernel(q_shift, exxdiv, kmesh);
    const auto& u_occ_q = u_occ_r_all_k[iq];

    if (symmetric_occ_square && iq == ik_target && n_trial == n_occ) {
      const int npairs = n_occ * (n_occ + 1) / 2;
      std::vector<Complex> pair_buf(static_cast<std::size_t>(npairs) * ngrid);
      int p = 0;
      for (int i = 0; i < n_occ; ++i) {
        const Complex* ui = u_occ_q.data() + static_cast<std::size_t>(i) * ngrid;
        for (int j = i; j < n_occ; ++j) {
          const Complex* uj = u_occ_q.data() + static_cast<std::size_t>(j) * ngrid;
          Complex* dst = pair_buf.data() + static_cast<std::size_t>(p) * ngrid;
          for (int ir = 0; ir < ngrid; ++ir) dst[ir] = std::conj(ui[ir]) * uj[ir];
          ++p;
        }
      }
      fft3d_batch_host(basis.fft_grid(), npairs, pair_buf.data(), pair_buf.data(), -1);
      for (int ip = 0; ip < npairs; ++ip) {
        Complex* box = pair_buf.data() + static_cast<std::size_t>(ip) * ngrid;
        for (int ig = 0; ig < ngrid; ++ig) box[ig] *= (coul_g[ig] * inv_ngrid);
      }
      fft3d_batch_host(basis.fft_grid(), npairs, pair_buf.data(), pair_buf.data(), +1);
      poisson_fft_counter += 2 * npairs;

      p = 0;
      for (int i = 0; i < n_occ; ++i) {
        const Complex* ui = u_occ_q.data() + static_cast<std::size_t>(i) * ngrid;
        Complex* wi = w_r.data() + static_cast<std::size_t>(i) * ngrid;
        for (int j = i; j < n_occ; ++j) {
          const Complex* phi_ij = pair_buf.data() + static_cast<std::size_t>(p) * ngrid;
          Complex* wj = w_r.data() + static_cast<std::size_t>(j) * ngrid;
          if (i == j) {
            for (int ir = 0; ir < ngrid; ++ir) wi[ir] -= wq * ui[ir] * phi_ij[ir];
          } else {
            const Complex* uj = u_occ_q.data() + static_cast<std::size_t>(j) * ngrid;
            for (int ir = 0; ir < ngrid; ++ir) {
              wj[ir] -= wq * ui[ir] * phi_ij[ir];
              wi[ir] -= wq * uj[ir] * std::conj(phi_ij[ir]);
            }
          }
          ++p;
        }
      }
    } else {
      const int npairs = n_occ * n_trial;
      std::vector<Complex> pair_buf(static_cast<std::size_t>(npairs) * ngrid);
      for (int i = 0; i < n_occ; ++i) {
        const Complex* ui = u_occ_q.data() + static_cast<std::size_t>(i) * ngrid;
        for (int b = 0; b < n_trial; ++b) {
          const Complex* ub = u_trial_r.data() + static_cast<std::size_t>(b) * ngrid;
          Complex* dst = pair_buf.data() + static_cast<std::size_t>(i * n_trial + b) * ngrid;
          for (int ir = 0; ir < ngrid; ++ir) dst[ir] = std::conj(ui[ir]) * ub[ir];
        }
      }
      fft3d_batch_host(basis.fft_grid(), npairs, pair_buf.data(), pair_buf.data(), -1);
      for (int ip = 0; ip < npairs; ++ip) {
        Complex* box = pair_buf.data() + static_cast<std::size_t>(ip) * ngrid;
        for (int ig = 0; ig < ngrid; ++ig) box[ig] *= (coul_g[ig] * inv_ngrid);
      }
      fft3d_batch_host(basis.fft_grid(), npairs, pair_buf.data(), pair_buf.data(), +1);
      poisson_fft_counter += 2 * npairs;

      for (int i = 0; i < n_occ; ++i) {
        const Complex* ui = u_occ_q.data() + static_cast<std::size_t>(i) * ngrid;
        for (int b = 0; b < n_trial; ++b) {
          const Complex* phi = pair_buf.data() + static_cast<std::size_t>(i * n_trial + b) * ngrid;
          Complex* wb = w_r.data() + static_cast<std::size_t>(b) * ngrid;
          for (int ir = 0; ir < ngrid; ++ir) wb[ir] -= wq * ui[ir] * phi[ir];
        }
      }
    }
  }

  // Transform W(r) back to G-space: c_W(G) = (sqrt(vol) / ngrid) * FFT_FWD(w_r)
  const double scale_back = std::sqrt(vol) * inv_ngrid;
  fft3d_batch_host(basis.fft_grid(), n_trial, w_r.data(), w_r.data(), -1);
  for (int b = 0; b < n_trial; ++b) {
    const Complex* box = w_r.data() + static_cast<std::size_t>(b) * ngrid;
    Complex* dst = vx_psi_out + static_cast<std::size_t>(b) * npw;
    for (int ig = 0; ig < npw; ++ig) {
      dst[ig] = scale_back * box[kp_t.fft_indices[ig]];
    }
  }
}

PwScfResult run_scf_host(const PwBasis& basis, const PwScfOptions& opts) {
  const auto t_start = std::chrono::steady_clock::now();
  const int n_elec = basis.cell().num_electrons(opts.charge);
  if (n_elec % 2 != 0) {
    throw std::invalid_argument("PwKohnShamSolver: closed-shell RKS requires even electron count");
  }
  const int n_occ = n_elec / 2;
  const int nkpts = basis.nkpts();
  const int ngrid = basis.ngrid();
  const double vol = basis.cell().volume();
  const double dvol = vol / static_cast<double>(ngrid);

  const auto xc_id = xc_gen::parse_xc_functional(opts.xc);
  const double alpha_x = xc_gen::xc_exact_exchange_fraction_by_id(xc_id);
  const bool is_hybrid = (std::abs(alpha_x) > 0.0);

  int min_npw = basis.kpoint(0).npw;
  for (int ik = 1; ik < nkpts; ++ik) min_npw = std::min(min_npw, basis.kpoint(ik).npw);

  const int nband_out =
      (opts.nbands > 0) ? std::min(min_npw, std::max(opts.nbands, n_occ))
                        : std::min(min_npw, std::max(n_occ + 2, n_occ));
  const int n_solve = std::min(min_npw, std::max(nband_out + 4, n_occ + 6));

  PwScfResult res;
  res.nbands = nband_out;
  res.ewald_energy = basis.cell().ewald_energy();

  const std::vector<double> vloc_ion_r = basis.compute_local_pp_grid();
  std::vector<NonLocalProjector> nl_projs(static_cast<std::size_t>(nkpts));
  for (int ik = 0; ik < nkpts; ++ik) {
    nl_projs[ik] = NonLocalProjector(basis.cell(), basis.kpoint(ik));
  }

  std::vector<std::vector<Complex>> psi_k(static_cast<std::size_t>(nkpts));
  std::vector<std::vector<double>> evals_k(static_cast<std::size_t>(nkpts));
  std::vector<double> rho_r = initial_atomic_gaussian_density(basis, n_elec);

  auto compute_variational_components = [&](const std::vector<double>& rho_eval,
                                            const std::vector<std::vector<Complex>>& orbs,
                                            xc_gen::XcFunctionalId eval_xc_id,
                                            double exx_raw_unscaled) {
    std::vector<Complex> rg;
    std::vector<double> vh_r, vxc_r;
    double eh = 0.0, exc = 0.0;
    eval_hartree_and_xc_host(basis, eval_xc_id, opts.rho_cutoff, rho_eval, rg, vh_r, vxc_r, eh, exc);

    double e_kin = 0.0;
    double e_nl = 0.0;
    for (int ik = 0; ik < nkpts; ++ik) {
      const auto& kp = basis.kpoint(ik);
      const double wk_occ = 2.0 * kp.weight;
      for (int b = 0; b < n_occ; ++b) {
        const Complex* col = orbs[ik].data() + static_cast<std::size_t>(b) * kp.npw;
        for (int ig = 0; ig < kp.npw; ++ig) {
          e_kin += wk_occ * kp.kin[ig] * std::norm(col[ig]);
        }
      }
      if (nl_projs[ik].nproj() > 0 && n_occ > 0) {
        std::vector<Complex> nl_psi(static_cast<std::size_t>(kp.npw) * n_occ, Complex(0.0, 0.0));
        nl_projs[ik].apply_add_host(orbs[ik].data(), nl_psi.data(), n_occ);
        for (int b = 0; b < n_occ; ++b) {
          const Complex* c1 = orbs[ik].data() + static_cast<std::size_t>(b) * kp.npw;
          const Complex* c2 = nl_psi.data() + static_cast<std::size_t>(b) * kp.npw;
          Complex dot(0.0, 0.0);
          for (int ig = 0; ig < kp.npw; ++ig) dot += std::conj(c1[ig]) * c2[ig];
          e_nl += wk_occ * dot.real();
        }
      }
    }
    double e_loc = 0.0;
    for (int ir = 0; ir < ngrid; ++ir) e_loc += rho_eval[ir] * vloc_ion_r[ir];
    e_loc *= dvol;

    res.kinetic_energy = e_kin;
    res.local_pp_energy = e_loc;
    res.nonlocal_pp_energy = e_nl;
    res.hartree_energy = eh;
    res.xc_energy = exc;
    res.exx_energy = alpha_x * exx_raw_unscaled;
    res.total_energy =
        res.ewald_energy + e_kin + e_loc + e_nl + eh + exc + res.exx_energy;
  };

  // Semi-local SCF driver (also used as inner fixed-ACE loop!)
  auto run_semilocal_or_fixed_ace_loop = [&](xc_gen::XcFunctionalId loop_xc_id,
                                             const std::vector<std::vector<Complex>>* ace_xi_k,
                                             double ace_alpha_x,
                                             double exx_const_raw,
                                             int max_iters,
                                             double e_tol,
                                             double d_tol) -> bool {
    KerkerPulayDiis diis(ngrid, basis.g2(), vol, opts.mixing_beta, opts.kerker_q0, opts.diis_max_vec);
    double prev_etot = 1e100;
    bool loop_conv = false;

    for (int iter = 0; iter < max_iters; ++iter) {
      ++res.scf_iterations;
      std::vector<Complex> rho_in_g;
      std::vector<double> vh_r, vxc_r;
      double eh_in = 0.0, exc_in = 0.0;
      eval_hartree_and_xc_host(
          basis, loop_xc_id, opts.rho_cutoff, rho_r, rho_in_g, vh_r, vxc_r, eh_in, exc_in);

      std::vector<double> veff_r(static_cast<std::size_t>(ngrid));
      for (int ir = 0; ir < ngrid; ++ir) {
        veff_r[ir] = vloc_ion_r[ir] + vh_r[ir] + vxc_r[ir];
      }

      const double eig_tol_iter =
          (iter == 0 && psi_k[0].empty())
              ? std::max(opts.eig_tol, 1e-7)
              : std::clamp(0.02 * std::max(res.final_density_rms, 1e-8), opts.eig_tol, 1e-5);

      for (int ik = 0; ik < nkpts; ++ik) {
        const Complex* xi_ptr =
            (ace_xi_k != nullptr && !(*ace_xi_k)[ik].empty()) ? (*ace_xi_k)[ik].data() : nullptr;
        auto apply_h = [&](const Complex* x, Complex* hx, int nb) {
          apply_pw_hamiltonian_local_nl_host(basis, ik, veff_r.data(), &nl_projs[ik],
                                             xi_ptr, (xi_ptr != nullptr) ? n_occ : 0, ace_alpha_x,
                                             x, hx, nb);
        };
        auto eig_res = run_eigensolver_host(
            basis.kpoint(ik), nband_out, n_solve, opts.eigensolver, eig_tol_iter,
            opts.max_eig_iter, apply_h, psi_k[ik]);
        res.total_eig_iterations += eig_res.iterations;
        res.total_hpsi_calls += eig_res.hpsi_calls;
        evals_k[ik] = std::move(eig_res.eigenvalues);
      }

      std::vector<double> rho_out_r;
      compute_density_host(basis, n_occ, psi_k, rho_out_r);

      std::vector<Complex> rho_out_g(static_cast<std::size_t>(ngrid));
      {
        std::vector<Complex> tmp_c(static_cast<std::size_t>(ngrid));
        for (int ir = 0; ir < ngrid; ++ir) tmp_c[ir] = Complex(rho_out_r[ir], 0.0);
        fft3d_batch_host(basis.fft_grid(), 1, tmp_c.data(), rho_out_g.data(), -1);
        const double inv_ng = 1.0 / static_cast<double>(ngrid);
        for (auto& c : rho_out_g) c *= inv_ng;
      }

      double exx_curr = exx_const_raw;
      if (ace_xi_k != nullptr && std::abs(ace_alpha_x) > 0.0 && n_occ > 0) {
        exx_curr = 0.0;
        for (int ik = 0; ik < nkpts; ++ik) {
          const int npw = basis.kpoint(ik).npw;
          std::vector<Complex> ov(static_cast<std::size_t>(n_occ) * n_occ);
          zgemm_host('C', 'N', n_occ, n_occ, npw, Complex(1.0, 0.0), (*ace_xi_k)[ik].data(), npw,
                     psi_k[ik].data(), npw, Complex(0.0, 0.0), ov.data(), n_occ);
          double fn2 = 0.0;
          for (const auto& z : ov) fn2 += std::norm(z);
          exx_curr -= basis.kpoint(ik).weight * fn2;
        }
      }

      compute_variational_components(rho_out_r, psi_k, loop_xc_id, exx_curr);
      const double de = std::abs(res.total_energy - prev_etot);
      prev_etot = res.total_energy;

      std::vector<double> rho_next_r;
      const double d_rms = diis.step(basis.fft_grid(), static_cast<double>(n_elec),
                                     rho_in_g, rho_out_g, rho_next_r);
      res.final_density_rms = d_rms;

      if (iter >= 1 && de < e_tol && d_rms < d_tol) {
        rho_r = std::move(rho_out_r);
        loop_conv = true;
        break;
      }
      rho_r = std::move(rho_next_r);
    }
    return loop_conv;
  };

  if (!is_hybrid) {
    res.converged = run_semilocal_or_fixed_ace_loop(
        xc_id, nullptr, 0.0, 0.0, opts.max_scf_iter, opts.conv_tol, opts.density_tol);
  } else if (opts.use_ace) {
    // Step 0: Warm-up orbitals with fast semi-local pre-SCF
    const auto pre_xc = (xc_id == xc_gen::XcFunctionalId::k_hf)
                            ? xc_gen::XcFunctionalId::k_svwn
                            : xc_gen::XcFunctionalId::k_pbe;
    run_semilocal_or_fixed_ace_loop(pre_xc, nullptr, 0.0, 0.0, 5, 1e-3, 2e-3);

    // Unified Kerker-DIIS + Lin (2016) Adaptive Compressed Exchange (ACE) SCF loop:
    // Rebuilds low-rank ACE projector Xi_k once per SCF step and applies -Xi_k (Xi_k^H Psi)
    // via pure ZGEMM with ZERO 3D Poisson FFTs across all inner Davidson/LOBPCG iterations!
    KerkerPulayDiis diis(ngrid, basis.g2(), vol, opts.mixing_beta, opts.kerker_q0, opts.diis_max_vec);
    std::vector<std::vector<Complex>> ace_xi_k(static_cast<std::size_t>(nkpts));
    double prev_etot = 1e100;
    const double inv_sqrt_vol = 1.0 / std::sqrt(vol);

    for (int iter = 0; iter < opts.max_scf_iter; ++iter) {
      ++res.scf_iterations;
      ++res.ace_iterations;

      std::vector<Complex> rho_in_g;
      std::vector<double> vh_r, vxc_r;
      double eh_in = 0.0, exc_in = 0.0;
      eval_hartree_and_xc_host(
          basis, xc_id, opts.rho_cutoff, rho_r, rho_in_g, vh_r, vxc_r, eh_in, exc_in);

      std::vector<double> veff_r(static_cast<std::size_t>(ngrid));
      for (int ir = 0; ir < ngrid; ++ir) veff_r[ir] = vloc_ion_r[ir] + vh_r[ir] + vxc_r[ir];

      // Build low-rank ACE projectors Xi_k = W_k * L_k^{-H} from current occupied orbitals
      std::vector<std::vector<Complex>> u_occ_r(static_cast<std::size_t>(nkpts));
      for (int iq = 0; iq < nkpts; ++iq) {
        const auto& kp = basis.kpoint(iq);
        u_occ_r[iq].assign(static_cast<std::size_t>(n_occ) * ngrid, Complex(0.0, 0.0));
        for (int j = 0; j < n_occ; ++j) {
          const Complex* src = psi_k[iq].data() + static_cast<std::size_t>(j) * kp.npw;
          Complex* box = u_occ_r[iq].data() + static_cast<std::size_t>(j) * ngrid;
          for (int ig = 0; ig < kp.npw; ++ig) box[kp.fft_indices[ig]] = src[ig];
        }
        fft3d_batch_host(basis.fft_grid(), n_occ, u_occ_r[iq].data(), u_occ_r[iq].data(), +1);
        for (auto& v : u_occ_r[iq]) v *= inv_sqrt_vol;
      }

      double exx_build = 0.0;
      for (int ik = 0; ik < nkpts; ++ik) {
        const int npw = basis.kpoint(ik).npw;
        std::vector<Complex> W(static_cast<std::size_t>(npw) * n_occ, Complex(0.0, 0.0));
        apply_exact_exchange_poisson_host(
            basis, ik, n_occ, u_occ_r, psi_k[ik].data(), n_occ,
            (nkpts == 1), opts.exxdiv, opts.kmesh, W.data(), res.total_poisson_ffts);

        std::vector<Complex> B(static_cast<std::size_t>(n_occ) * n_occ);
        zgemm_host('C', 'N', n_occ, n_occ, npw, Complex(-1.0, 0.0), psi_k[ik].data(), npw,
                   W.data(), npw, Complex(0.0, 0.0), B.data(), n_occ);
        double tr_b = 0.0;
        for (int j = 0; j < n_occ; ++j) tr_b += B[j + static_cast<std::size_t>(j) * n_occ].real();
        exx_build -= basis.kpoint(ik).weight * tr_b;

        ace_xi_k[ik].resize(static_cast<std::size_t>(npw) * n_occ);
        cholesky_and_trsm_ace_host(npw, n_occ, B.data(), W.data(), ace_xi_k[ik].data());
      }

      // Inner Davidson/LOBPCG uses pure ZGEMM ACE operator (ZERO 3D Poisson FFTs!)
      std::vector<std::vector<Complex>> psi_occ_prev(static_cast<std::size_t>(nkpts));
      for (int ik = 0; ik < nkpts; ++ik) {
        const int npw = basis.kpoint(ik).npw;
        psi_occ_prev[ik].assign(
            psi_k[ik].begin(),
            psi_k[ik].begin() + static_cast<std::size_t>(npw) * n_occ);
      }

      const double eig_tol_iter =
          std::clamp(0.02 * std::max(res.final_density_rms, 1e-8), opts.eig_tol, 1e-5);
      for (int ik = 0; ik < nkpts; ++ik) {
        auto apply_h = [&](const Complex* x, Complex* hx, int nb) {
          apply_pw_hamiltonian_local_nl_host(
              basis, ik, veff_r.data(), &nl_projs[ik],
              ace_xi_k[ik].data(), n_occ, alpha_x, x, hx, nb);
        };
        auto eig_res = run_eigensolver_host(
            basis.kpoint(ik), nband_out, n_solve, opts.eigensolver, eig_tol_iter,
            opts.max_eig_iter, apply_h, psi_k[ik]);
        res.total_eig_iterations += eig_res.iterations;
        res.total_hpsi_calls += eig_res.hpsi_calls;
        evals_k[ik] = std::move(eig_res.eigenvalues);
      }

      double exx_proj = 0.0;
      double subspace_err2 = 0.0;
      for (int ik = 0; ik < nkpts; ++ik) {
        const int npw = basis.kpoint(ik).npw;
        std::vector<Complex> ov(static_cast<std::size_t>(n_occ) * n_occ);
        zgemm_host('C', 'N', n_occ, n_occ, npw, Complex(1.0, 0.0), ace_xi_k[ik].data(), npw,
                   psi_k[ik].data(), npw, Complex(0.0, 0.0), ov.data(), n_occ);
        double fn2 = 0.0;
        for (const auto& z : ov) fn2 += std::norm(z);
        exx_proj -= basis.kpoint(ik).weight * fn2;

        // Second-order Grassmann subspace stationarity ||(I - P_prev) Psi_occ||_F^2
        zgemm_host('C', 'N', n_occ, n_occ, npw, Complex(1.0, 0.0), psi_occ_prev[ik].data(), npw,
                   psi_k[ik].data(), npw, Complex(0.0, 0.0), ov.data(), n_occ);
        double proj_norm2 = 0.0;
        for (const auto& z : ov) proj_norm2 += std::norm(z);
        subspace_err2 += basis.kpoint(ik).weight *
                         std::max(0.0, static_cast<double>(n_occ) - proj_norm2);
      }
      (void)exx_build;

      std::vector<double> rho_out_r;
      compute_density_host(basis, n_occ, psi_k, rho_out_r);
      compute_variational_components(rho_out_r, psi_k, xc_id, exx_proj);

      std::vector<Complex> rho_out_g(static_cast<std::size_t>(ngrid));
      {
        std::vector<Complex> tmp_c(static_cast<std::size_t>(ngrid));
        for (int ir = 0; ir < ngrid; ++ir) tmp_c[ir] = Complex(rho_out_r[ir], 0.0);
        fft3d_batch_host(basis.fft_grid(), 1, tmp_c.data(), rho_out_g.data(), -1);
        const double inv_ng = 1.0 / static_cast<double>(ngrid);
        for (auto& c : rho_out_g) c *= inv_ng;
      }

      const double de = std::abs(res.total_energy - prev_etot);
      prev_etot = res.total_energy;

      std::vector<double> rho_next_r;
      const double d_rms = diis.step(basis.fft_grid(), static_cast<double>(n_elec),
                                     rho_in_g, rho_out_g, rho_next_r);
      res.final_density_rms = d_rms;

      if (iter >= 2 && de < opts.conv_tol && subspace_err2 < opts.conv_tol &&
          d_rms < opts.density_tol) {
        rho_r = std::move(rho_out_r);
        res.converged = true;
        break;
      }
      rho_r = std::move(rho_next_r);
    }
  } else {
    // Uncompressed Exact Exchange reference (evaluates Poisson FFTs inside EVERY H*psi call!)
    const auto pre_xc = (xc_id == xc_gen::XcFunctionalId::k_hf)
                            ? xc_gen::XcFunctionalId::k_svwn
                            : xc_gen::XcFunctionalId::k_pbe;
    run_semilocal_or_fixed_ace_loop(pre_xc, nullptr, 0.0, 0.0, 6, 1e-4, 1e-3);

    KerkerPulayDiis diis(ngrid, basis.g2(), vol, opts.mixing_beta, opts.kerker_q0, opts.diis_max_vec);
    double prev_etot = 1e100;
    const double inv_sqrt_vol = 1.0 / std::sqrt(vol);

    for (int iter = 0; iter < opts.max_scf_iter; ++iter) {
      ++res.scf_iterations;
      std::vector<Complex> rho_in_g;
      std::vector<double> vh_r, vxc_r;
      double eh_in = 0.0, exc_in = 0.0;
      eval_hartree_and_xc_host(
          basis, xc_id, opts.rho_cutoff, rho_r, rho_in_g, vh_r, vxc_r, eh_in, exc_in);

      std::vector<double> veff_r(static_cast<std::size_t>(ngrid));
      for (int ir = 0; ir < ngrid; ++ir) veff_r[ir] = vloc_ion_r[ir] + vh_r[ir] + vxc_r[ir];

      std::vector<std::vector<Complex>> u_occ_r(static_cast<std::size_t>(nkpts));
      for (int iq = 0; iq < nkpts; ++iq) {
        const auto& kp = basis.kpoint(iq);
        u_occ_r[iq].assign(static_cast<std::size_t>(n_occ) * ngrid, Complex(0.0, 0.0));
        for (int j = 0; j < n_occ; ++j) {
          const Complex* src = psi_k[iq].data() + static_cast<std::size_t>(j) * kp.npw;
          Complex* box = u_occ_r[iq].data() + static_cast<std::size_t>(j) * ngrid;
          for (int ig = 0; ig < kp.npw; ++ig) box[kp.fft_indices[ig]] = src[ig];
        }
        fft3d_batch_host(basis.fft_grid(), n_occ, u_occ_r[iq].data(), u_occ_r[iq].data(), +1);
        for (auto& v : u_occ_r[iq]) v *= inv_sqrt_vol;
      }

      const double eig_tol_iter =
          std::clamp(0.02 * std::max(res.final_density_rms, 1e-8), opts.eig_tol, 1e-5);
      for (int ik = 0; ik < nkpts; ++ik) {
        const int npw = basis.kpoint(ik).npw;
        auto apply_h = [&](const Complex* x, Complex* hx, int nb) {
          apply_pw_hamiltonian_local_nl_host(
              basis, ik, veff_r.data(), &nl_projs[ik], nullptr, 0, 0.0, x, hx, nb);
          std::vector<Complex> vx_x(static_cast<std::size_t>(npw) * nb);
          apply_exact_exchange_poisson_host(
              basis, ik, n_occ, u_occ_r, x, nb, false, opts.exxdiv, opts.kmesh,
              vx_x.data(), res.total_poisson_ffts);
          for (std::size_t idx = 0; idx < vx_x.size(); ++idx) hx[idx] += alpha_x * vx_x[idx];
        };
        auto eig_res = run_eigensolver_host(
            basis.kpoint(ik), nband_out, n_solve, opts.eigensolver, eig_tol_iter,
            opts.max_eig_iter, apply_h, psi_k[ik]);
        res.total_eig_iterations += eig_res.iterations;
        res.total_hpsi_calls += eig_res.hpsi_calls;
        evals_k[ik] = std::move(eig_res.eigenvalues);
      }

      // Compute exact exchange energy on updated orbitals
      for (int iq = 0; iq < nkpts; ++iq) {
        const auto& kp = basis.kpoint(iq);
        u_occ_r[iq].assign(static_cast<std::size_t>(n_occ) * ngrid, Complex(0.0, 0.0));
        for (int j = 0; j < n_occ; ++j) {
          const Complex* src = psi_k[iq].data() + static_cast<std::size_t>(j) * kp.npw;
          Complex* box = u_occ_r[iq].data() + static_cast<std::size_t>(j) * ngrid;
          for (int ig = 0; ig < kp.npw; ++ig) box[kp.fft_indices[ig]] = src[ig];
        }
        fft3d_batch_host(basis.fft_grid(), n_occ, u_occ_r[iq].data(), u_occ_r[iq].data(), +1);
        for (auto& v : u_occ_r[iq]) v *= inv_sqrt_vol;
      }
      double exx_raw = 0.0;
      for (int ik = 0; ik < nkpts; ++ik) {
        const int npw = basis.kpoint(ik).npw;
        std::vector<Complex> W(static_cast<std::size_t>(npw) * n_occ);
        apply_exact_exchange_poisson_host(
            basis, ik, n_occ, u_occ_r, psi_k[ik].data(), n_occ, (nkpts == 1),
            opts.exxdiv, opts.kmesh, W.data(), res.total_poisson_ffts);
        for (int j = 0; j < n_occ; ++j) {
          const Complex* pcol = psi_k[ik].data() + static_cast<std::size_t>(j) * npw;
          const Complex* wcol = W.data() + static_cast<std::size_t>(j) * npw;
          Complex dot(0.0, 0.0);
          for (int ig = 0; ig < npw; ++ig) dot += std::conj(pcol[ig]) * wcol[ig];
          exx_raw += basis.kpoint(ik).weight * dot.real();
        }
      }

      std::vector<double> rho_out_r;
      compute_density_host(basis, n_occ, psi_k, rho_out_r);
      compute_variational_components(rho_out_r, psi_k, xc_id, exx_raw);

      std::vector<Complex> rho_out_g(static_cast<std::size_t>(ngrid));
      {
        std::vector<Complex> tmp_c(static_cast<std::size_t>(ngrid));
        for (int ir = 0; ir < ngrid; ++ir) tmp_c[ir] = Complex(rho_out_r[ir], 0.0);
        fft3d_batch_host(basis.fft_grid(), 1, tmp_c.data(), rho_out_g.data(), -1);
        const double inv_ng = 1.0 / static_cast<double>(ngrid);
        for (auto& c : rho_out_g) c *= inv_ng;
      }

      const double de = std::abs(res.total_energy - prev_etot);
      prev_etot = res.total_energy;

      std::vector<double> rho_next_r;
      const double d_rms = diis.step(basis.fft_grid(), static_cast<double>(n_elec),
                                     rho_in_g, rho_out_g, rho_next_r);
      res.final_density_rms = d_rms;
      if (iter >= 1 && de < opts.conv_tol && d_rms < opts.density_tol) {
        rho_r = std::move(rho_out_r);
        res.converged = true;
        break;
      }
      rho_r = std::move(rho_next_r);
    }
  }

  // Final HOMO / LUMO extraction across k-points
  double homo = -1e100;
  double lumo = +1e100;
  for (int ik = 0; ik < nkpts; ++ik) {
    if (n_occ > 0 && static_cast<int>(evals_k[ik].size()) >= n_occ) {
      homo = std::max(homo, evals_k[ik][n_occ - 1]);
    }
    if (static_cast<int>(evals_k[ik].size()) > n_occ) {
      lumo = std::min(lumo, evals_k[ik][n_occ]);
    }
  }
  res.homo_energy = (homo > -1e90) ? homo : 0.0;
  res.lumo_energy = (lumo < +1e90) ? lumo : 0.0;
  res.eigenvalues = std::move(evals_k);
  res.density_r = std::move(rho_r);

  const auto t_end = std::chrono::steady_clock::now();
  res.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  return res;
}

}  // namespace

PwKohnShamSolver::PwKohnShamSolver(const PwBasis& basis, PwScfOptions opts)
    : basis_(basis), opts_(std::move(opts)) {}

PwKohnShamSolver::~PwKohnShamSolver() = default;

PwScfResult PwKohnShamSolver::run() {
  if (opts_.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::pw_cuda_device_count() == 0) {
      throw std::runtime_error("PwKohnShamSolver::run: no CUDA device available");
    }
    return detail::pw_cuda_run_scf(basis_, opts_);
#else
    throw std::runtime_error("PwKohnShamSolver::run: gpu4dft was built without CUDA");
#endif
  }
  if (opts_.backend != Backend::Host) {
    throw std::runtime_error("PwKohnShamSolver::run: unsupported backend");
  }
  return run_scf_host(basis_, opts_);
}

}  // namespace gpu4dft::pw

// ============================================================================
// C-ABI Entry Points (`gpu4dft_pw_c_*`) for Standalone & Pytest Verification
// ============================================================================

namespace {

thread_local std::string g_last_pw_error;

gpu4dft::Backend decode_backend(int backend_id) {
  if (backend_id == 0) return gpu4dft::Backend::Host;
  if (backend_id == 1) return gpu4dft::Backend::Cuda;
  throw std::invalid_argument("Invalid backend_id (0=Host, 1=Cuda)");
}

gpu4dft::pw::PwSolverType decode_solver(int solver_id) {
  if (solver_id == 0) return gpu4dft::pw::PwSolverType::Davidson;
  if (solver_id == 1) return gpu4dft::pw::PwSolverType::LOBPCG;
  throw std::invalid_argument("Invalid solver_id (0=Davidson, 1=LOBPCG)");
}

std::array<std::array<double, 3>, 3> unpack_lattice(const double* a9) {
  std::array<std::array<double, 3>, 3> a{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) a[i][j] = a9[3 * i + j];
  }
  return a;
}

std::vector<gpu4dft::pw::PwAtom> unpack_atoms(int natom,
                                              const int* atomic_numbers,
                                              const double* atom_pos_3n) {
  std::vector<gpu4dft::pw::PwAtom> atoms(static_cast<std::size_t>(natom));
  for (int i = 0; i < natom; ++i) {
    atoms[i].atomic_number = atomic_numbers[i];
    atoms[i].pos = {atom_pos_3n[3 * i + 0], atom_pos_3n[3 * i + 1], atom_pos_3n[3 * i + 2]};
  }
  return atoms;
}

}  // namespace

extern "C" {

const char* gpu4dft_pw_c_last_error() { return g_last_pw_error.c_str(); }

int gpu4dft_pw_c_cuda_available() { return gpu4dft::pw::cuda_pw_available() ? 1 : 0; }

int gpu4dft_pw_c_ewald(const double* a_vecs_9,
                       int natom,
                       const int* atomic_numbers,
                       const double* atom_pos_3n,
                       const char* pseudo_family,
                       const int* kmesh_3,
                       double* out_ewald,
                       double* out_madelung) {
  try {
    const auto a = unpack_lattice(a_vecs_9);
    const auto atoms = unpack_atoms(natom, atomic_numbers, atom_pos_3n);
    const std::string fam = (pseudo_family != nullptr) ? pseudo_family : "gth-pade";
    gpu4dft::pw::PwCell cell(a, atoms, fam);
    if (out_ewald != nullptr) *out_ewald = cell.ewald_energy();
    if (out_madelung != nullptr) {
      std::array<int, 3> km = {1, 1, 1};
      if (kmesh_3 != nullptr) km = {kmesh_3[0], kmesh_3[1], kmesh_3[2]};
      *out_madelung = cell.madelung_constant(km);
    }
    return 0;
  } catch (const std::exception& e) {
    g_last_pw_error = e.what();
    return -1;
  }
}

int gpu4dft_pw_c_fft3d(int n1,
                       int n2,
                       int n3,
                       int nbatch,
                       int direction,
                       int backend_id,
                       const double* in_re_im,
                       double* out_re_im) {
  try {
    const auto* c_in = reinterpret_cast<const gpu4dft::pw::Complex*>(in_re_im);
    auto* c_out = reinterpret_cast<gpu4dft::pw::Complex*>(out_re_im);
    gpu4dft::pw::fft3d_batch({n1, n2, n3}, nbatch, c_in, c_out, direction, decode_backend(backend_id));
    return 0;
  } catch (const std::exception& e) {
    g_last_pw_error = e.what();
    return -1;
  }
}

int gpu4dft_pw_c_basis_info(const double* a_vecs_9,
                            double ecut_wfc,
                            double ecut_rho,
                            const double* kpt_3,
                            const int* fft_grid_in_3,
                            int* fft_grid_out_3,
                            int* out_npw) {
  try {
    gpu4dft::pw::PwCell cell(unpack_lattice(a_vecs_9), {}, "none");
    std::array<double, 3> kpt = {0.0, 0.0, 0.0};
    if (kpt_3 != nullptr) kpt = {kpt_3[0], kpt_3[1], kpt_3[2]};
    std::array<int, 3> grid_in = {0, 0, 0};
    if (fft_grid_in_3 != nullptr) grid_in = {fft_grid_in_3[0], fft_grid_in_3[1], fft_grid_in_3[2]};
    const std::array<std::array<double, 3>, 1> kpts_arr = {kpt};
    const std::array<double, 1> kw_arr = {1.0};
    gpu4dft::pw::PwBasis basis(cell, ecut_wfc, ecut_rho, kpts_arr, kw_arr, grid_in);
    const auto fg = basis.fft_grid();
    if (fft_grid_out_3 != nullptr) {
      fft_grid_out_3[0] = fg[0];
      fft_grid_out_3[1] = fg[1];
      fft_grid_out_3[2] = fg[2];
    }
    if (out_npw != nullptr) *out_npw = basis.kpoint(0).npw;
    return 0;
  } catch (const std::exception& e) {
    g_last_pw_error = e.what();
    return -1;
  }
}

int gpu4dft_pw_c_solve_eigen(const double* a_vecs_9,
                             double ecut_wfc,
                             const double* kpt_3,
                             const int* fft_grid_3,
                             const double* vloc_r,
                             int nband,
                             int solver_id,
                             int backend_id,
                             double residual_tol,
                             int max_iter,
                             double* out_evals,
                             int* out_iters) {
  try {
    gpu4dft::pw::PwCell cell(unpack_lattice(a_vecs_9), {}, "none");
    std::array<double, 3> kpt = {0.0, 0.0, 0.0};
    if (kpt_3 != nullptr) kpt = {kpt_3[0], kpt_3[1], kpt_3[2]};
    std::array<int, 3> grid_in = {0, 0, 0};
    if (fft_grid_3 != nullptr) grid_in = {fft_grid_3[0], fft_grid_3[1], fft_grid_3[2]};
    const std::array<std::array<double, 3>, 1> kpts_arr = {kpt};
    const std::array<double, 1> kw_arr = {1.0};
    gpu4dft::pw::PwBasis basis(cell, ecut_wfc, 0.0, kpts_arr, kw_arr, grid_in);

    std::vector<double> v_vec;
    std::span<const double> v_span;
    if (vloc_r != nullptr) {
      v_span = std::span<const double>(vloc_r, static_cast<std::size_t>(basis.ngrid()));
    } else {
      v_vec.assign(static_cast<std::size_t>(basis.ngrid()), 0.0);
      v_span = v_vec;
    }

    auto res = gpu4dft::pw::solve_pw_eigenproblem(
        basis, 0, v_span, nband, decode_solver(solver_id), decode_backend(backend_id),
        residual_tol, max_iter, false);
    if (out_evals != nullptr) {
      std::copy(res.eigenvalues.begin(), res.eigenvalues.end(), out_evals);
    }
    if (out_iters != nullptr) *out_iters = res.iterations;
    return 0;
  } catch (const std::exception& e) {
    g_last_pw_error = e.what();
    return -1;
  }
}

int gpu4dft_pw_c_run_scf(const double* a_vecs_9,
                         int natom,
                         const int* atomic_numbers,
                         const double* atom_pos_3n,
                         const char* pseudo_family,
                         const char* xc_name,
                         double ecut_wfc,
                         double ecut_rho,
                         const int* fft_grid_in_3,
                         int nkpts,
                         const double* kpts_3nk,
                         const double* kweights_nk,
                         const int* kmesh_3,
                         int charge,
                         int nbands,
                         int solver_id,
                         int backend_id,
                         int exxdiv_mode,
                         int use_ace,
                         int max_scf_iter,
                         double conv_tol,
                         double* out_metrics_18,
                         double* out_eigenvalues,
                         int max_out_eigs) {
  try {
    const auto a = unpack_lattice(a_vecs_9);
    const auto atoms = unpack_atoms(natom, atomic_numbers, atom_pos_3n);
    const std::string fam = (pseudo_family != nullptr) ? pseudo_family : "gth-pade";
    gpu4dft::pw::PwCell cell(a, atoms, fam);

    std::vector<std::array<double, 3>> kpts;
    std::vector<double> kw;
    if (nkpts > 0 && kpts_3nk != nullptr) {
      kpts.resize(static_cast<std::size_t>(nkpts));
      for (int ik = 0; ik < nkpts; ++ik) {
        kpts[ik] = {kpts_3nk[3 * ik + 0], kpts_3nk[3 * ik + 1], kpts_3nk[3 * ik + 2]};
      }
      if (kweights_nk != nullptr) {
        kw.assign(kweights_nk, kweights_nk + nkpts);
      } else {
        kw.assign(static_cast<std::size_t>(nkpts), 1.0 / static_cast<double>(nkpts));
      }
    }

    std::array<int, 3> grid_in = {0, 0, 0};
    if (fft_grid_in_3 != nullptr) grid_in = {fft_grid_in_3[0], fft_grid_in_3[1], fft_grid_in_3[2]};

    gpu4dft::pw::PwBasis basis(cell, ecut_wfc, ecut_rho, kpts, kw, grid_in);

    gpu4dft::pw::PwScfOptions opts;
    opts.backend = decode_backend(backend_id);
    opts.eigensolver = decode_solver(solver_id);
    opts.xc = (xc_name != nullptr) ? xc_name : "lda";
    opts.exxdiv = (exxdiv_mode == 0) ? gpu4dft::pw::ExxDiv::None : gpu4dft::pw::ExxDiv::Ewald;
    if (kmesh_3 != nullptr) opts.kmesh = {kmesh_3[0], kmesh_3[1], kmesh_3[2]};
    opts.use_ace = (use_ace != 0);
    opts.charge = charge;
    opts.nbands = nbands;
    if (max_scf_iter > 0) opts.max_scf_iter = max_scf_iter;
    if (conv_tol > 0.0) opts.conv_tol = conv_tol;

    gpu4dft::pw::PwKohnShamSolver solver(basis, opts);
    const auto res = solver.run();

    if (out_metrics_18 != nullptr) {
      out_metrics_18[0] = res.total_energy;
      out_metrics_18[1] = res.kinetic_energy;
      out_metrics_18[2] = res.local_pp_energy;
      out_metrics_18[3] = res.nonlocal_pp_energy;
      out_metrics_18[4] = res.hartree_energy;
      out_metrics_18[5] = res.xc_energy;
      out_metrics_18[6] = res.exx_energy;
      out_metrics_18[7] = res.ewald_energy;
      out_metrics_18[8] = res.homo_energy;
      out_metrics_18[9] = res.lumo_energy;
      out_metrics_18[10] = res.final_density_rms;
      out_metrics_18[11] = res.converged ? 1.0 : 0.0;
      out_metrics_18[12] = static_cast<double>(res.scf_iterations);
      out_metrics_18[13] = static_cast<double>(res.ace_iterations);
      out_metrics_18[14] = static_cast<double>(res.total_eig_iterations);
      out_metrics_18[15] = static_cast<double>(res.total_hpsi_calls);
      out_metrics_18[16] = static_cast<double>(res.total_poisson_ffts);
      out_metrics_18[17] = res.elapsed_ms;
    }

    if (out_eigenvalues != nullptr && max_out_eigs > 0) {
      int pos = 0;
      for (const auto& kv : res.eigenvalues) {
        for (double ev : kv) {
          if (pos < max_out_eigs) out_eigenvalues[pos++] = ev;
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    g_last_pw_error = e.what();
    return -1;
  }
}

}  // extern "C"
