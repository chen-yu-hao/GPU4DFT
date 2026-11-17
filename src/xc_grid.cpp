#include "gpu4dft/xc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu4dft {

namespace detail {
#if defined(GPU4DFT_HAVE_CUDA)
int xc_cuda_device_count();
void compute_partition_weights_cuda(const double* raw_coords,
                                    const int* parent_atom,
                                    std::size_t n_raw,
                                    const double* atom_coords,
                                    const double* bragg_radii,
                                    int natom,
                                    PartitionScheme scheme,
                                    bool radii_adjust,
                                    double* weights_inout);
#endif
}  // namespace detail

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct OhOrbit {
  double a;
  double b;
  double c;
  double weight;  // Point weight on unit sphere (sum over all points == 4 * pi)
};

// Exact IEEE-754 octahedral (Oh) symmetry orbits for Lebedev angular grids.
constexpr OhOrbit kLebedev14[] = {
    {1.0, 0.0, 0.0, 0.8377580409572781},
    {0.5773502691896258, 0.5773502691896258, 0.5773502691896258, 0.9424777960769379},
};
constexpr OhOrbit kLebedev26[] = {
    {1.0, 0.0, 0.0, 0.5983986006837702},
    {0.7071067811865475, 0.7071067811865475, 0.0, 0.47871888054701617},
    {0.5773502691896258, 0.5773502691896258, 0.5773502691896258, 0.4039190554615448},
};
constexpr OhOrbit kLebedev50[] = {
    {1.0, 0.0, 0.0, 0.1595729601823387},
    {0.7071067811865475, 0.7071067811865475, 0.0, 0.2836852625463799},
    {0.5773502691896258, 0.5773502691896258, 0.5773502691896258, 0.2650718801466388},
    {0.9045340337332909, 0.3015113445777636, 0.3015113445777636, 0.2535056108973113},
};
constexpr OhOrbit kLebedev110[] = {
    {1.0, 0.0, 0.0, 0.04810746585139659},
    {0.5773502691896258, 0.5773502691896258, 0.5773502691896258, 0.12307173528167017},
    {0.9651240350865941, 0.18511563534473632, 0.18511563534473632, 0.10319173408833042},
    {0.6904210483822922, 0.6904210483822922, 0.21595729184584844, 0.1249450968725133},
    {0.8287699812525923, 0.3956894730559419, 0.3956894730559419, 0.12058024902852789},
    {0.8781589106040661, 0.4783690288121502, 0.0, 0.12183091738552138},
};
constexpr OhOrbit kLebedev194[] = {
    {1.0, 0.0, 0.0, 0.022397550621038462},
    {0.7071067811865475, 0.7071067811865475, 0.0, 0.07184075893484734},
    {0.5773502691896258, 0.5773502691896258, 0.5773502691896258, 0.07003719860124849},
    {0.6712973442695226, 0.6712973442695226, 0.3141969941825863, 0.07048105416807013},
    {0.9125090968674737, 0.2892465627575438, 0.2892465627575438, 0.06482032680351046},
    {0.7774932193147671, 0.4446933178717437, 0.4446933178717437, 0.069350927593711},
    {0.9829723027072532, 0.12993354476500701, 0.12993354476500701, 0.05160728216651316},
    {0.9383192181375916, 0.3457702197611283, 0.0, 0.06348336993464156},
    {0.8360360154824589, 0.525118572443642, 0.159041710538353, 0.06949515747104322},
};
constexpr OhOrbit kLebedev302[] = {
    {1.0, 0.0, 0.0, 0.010739109397555789},
    {0.5773502691896258, 0.5773502691896258, 0.5773502691896258, 0.04522786682091873},
    {0.8676436245440834, 0.3515640345570104, 0.3515640345570104, 0.043351319880953885},
    {0.6566329410219612, 0.6566329410219612, 0.37103417838482095, 0.045299536808460585},
    {0.7434520429875557, 0.4729054132581005, 0.4729054132581005, 0.04494651051683867},
    {0.9907056213794081, 0.096183085226148, 0.096183085226148, 0.029557378086976182},
    {0.9494543172264431, 0.22196452362941776, 0.22196452362941776, 0.0390682571589194},
    {0.7011766416089545, 0.7011766416089545, 0.12923867271051442, 0.04586782837866035},
    {0.964408914879206, 0.2644152887060663, 0.0, 0.03747725210708425},
    {0.8203264198277593, 0.5718955891878961, 0.0, 0.045249250350174325},
    {0.8000727494073951, 0.5448677372580774, 0.2510034751770465, 0.044881302269213164},
    {0.9024425295330004, 0.4127724083168531, 0.1233548532583327, 0.0426290524077215},
};

void append_oh_orbit(const OhOrbit& orb, std::vector<double>& xyz, std::vector<double>& w) {
  const std::array<std::array<double, 3>, 6> perms = {{
      {orb.a, orb.b, orb.c},
      {orb.a, orb.c, orb.b},
      {orb.b, orb.a, orb.c},
      {orb.b, orb.c, orb.a},
      {orb.c, orb.a, orb.b},
      {orb.c, orb.b, orb.a},
  }};
  std::vector<std::array<double, 3>> generated;
  generated.reserve(48);

  for (const auto& p : perms) {
    for (int sx : {-1, 1}) {
      if (p[0] == 0.0 && sx < 0) continue;
      for (int sy : {-1, 1}) {
        if (p[1] == 0.0 && sy < 0) continue;
        for (int sz : {-1, 1}) {
          if (p[2] == 0.0 && sz < 0) continue;
          const std::array<double, 3> cand = {
              sx * p[0], sy * p[1], sz * p[2]
          };
          bool dup = false;
          for (const auto& prev : generated) {
            const double dx = cand[0] - prev[0];
            const double dy = cand[1] - prev[1];
            const double dz = cand[2] - prev[2];
            if (dx * dx + dy * dy + dz * dz < 1e-24) {
              dup = true;
              break;
            }
          }
          if (!dup) generated.push_back(cand);
        }
      }
    }
  }

  for (const auto& pt : generated) {
    xyz.push_back(pt[0]);
    xyz.push_back(pt[1]);
    xyz.push_back(pt[2]);
    w.push_back(orb.weight);
  }
}

// Treutler & Ahlrichs (J. Chem. Phys. 102, 346, 1995) Table I atomic scaling radii (Bohr).
double treutler_radius_bohr(int z) {
  static constexpr double kXi[] = {
      1.0,  // Z=0 (ghost)
      0.8, 0.9,                                            // H, He
      1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,              // Li..Ne
      1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,              // Na..Ar
      2.2, 2.0, 1.6, 1.5, 1.4, 1.4, 1.4, 1.4, 1.3, 1.3,    // K..Ni
      1.3, 1.3, 1.2, 1.2, 1.2, 1.1, 1.1, 1.0,              // Cu..Kr
      2.3, 2.1, 1.8, 1.6, 1.5, 1.5, 1.4, 1.4, 1.4, 1.4,    // Rb..Pd
      1.5, 1.5, 1.5, 1.4, 1.4, 1.4, 1.3, 1.3               // Ag..Xe
  };
  if (z >= 0 && z < static_cast<int>(sizeof(kXi) / sizeof(kXi[0]))) return kXi[z];
  return 1.5;
}

// Bragg-Slater covalent radii (Angstrom -> converted to Bohr; only ratios matter in Becke/SSF!).
double bragg_slater_radius(int z) {
  // Bragg-Slater radii in Angstrom (Becke 1988, JCP 88, 2547; H = 0.35 as recommended by Becke)
  static constexpr double kRadiiAng[] = {
      0.75,  // Z=0
      0.35, 0.30,                                                  // H, He
      1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45,              // Li..Ne
      1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 0.95,              // Na..Ar
      2.20, 1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35,  // K..Ni
      1.35, 1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 1.10,              // Cu..Kr
      2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40,  // Rb..Pd
      1.60, 1.55, 1.55, 1.45, 1.45, 1.40, 1.40, 1.35               // Ag..Xe
  };
  constexpr double kAngToBohr = 1.8897261246257702;
  if (z >= 0 && z < static_cast<int>(sizeof(kRadiiAng) / sizeof(kRadiiAng[0]))) {
    return kRadiiAng[z] * kAngToBohr;
  }
  return 1.50 * kAngToBohr;
}

int default_n_rad(int z, int level) {
  const int lvl = std::clamp(level, 0, 5);
  // Row 1 (H, He), Row 2 (Li..Ne), Row 3+ (Na..)
  static constexpr int kRadH[6]   = {30, 40, 50, 60, 75, 99};
  static constexpr int kRadR2[6]  = {40, 50, 65, 75, 99, 125};
  static constexpr int kRadR3[6]  = {50, 65, 80, 99, 120, 150};
  if (z <= 2) return kRadH[lvl];
  if (z <= 10) return kRadR2[lvl];
  return kRadR3[lvl];
}

int default_max_ang(int level) {
  const int lvl = std::clamp(level, 0, 5);
  static constexpr int kAng[6] = {50, 110, 194, 302, 302, 302};
  return kAng[lvl];
}

int pruned_lebedev_size(double r, double xi_z, int max_ang) {
  // NWChem / Treutler-style 5-zone radial angular pruning
  const double ratio = r / xi_z;
  int target = max_ang;
  if (ratio < 0.15) {
    target = 14;
  } else if (ratio < 0.40) {
    target = 50;
  } else if (ratio < 0.80) {
    target = 110;
  } else if (ratio < 3.50) {
    target = max_ang;
  } else {
    target = 194;
  }
  return std::min(target, max_ang);
}

inline double step_becke(double nu) {
  if (nu <= -1.0) return 1.0;
  if (nu >= 1.0) return 0.0;
  double f = nu;
  for (int k = 0; k < 3; ++k) {
    f = 0.5 * f * (3.0 - f * f);
  }
  return 0.5 * (1.0 - f);
}

inline double step_ssf(double nu) {
  constexpr double kA = 0.64;
  if (nu <= -kA) return 1.0;
  if (nu >= kA) return 0.0;
  const double z = nu / kA;
  const double z2 = z * z;
  const double g = (1.0 / 16.0) * z * (35.0 + z2 * (-35.0 + z2 * (21.0 - 5.0 * z2)));
  return 0.5 * (1.0 - g);
}

void compute_partition_weights_host(const std::vector<double>& raw_coords,
                                    const std::vector<int>& parent_atom,
                                    const std::vector<double>& atom_coords,
                                    const std::vector<double>& bragg_radii,
                                    PartitionScheme scheme,
                                    bool radii_adjust,
                                    std::vector<double>& raw_weights) {
  const std::size_t n_raw = raw_weights.size();
  const int natom = static_cast<int>(bragg_radii.size());
  if (natom <= 1 || n_raw == 0) return;

  // Precompute pairwise 1 / R_BC and heteronuclear size adjustment a_BC
  std::vector<double> inv_rij(static_cast<std::size_t>(natom * natom), 0.0);
  std::vector<double> a_ij(static_cast<std::size_t>(natom * natom), 0.0);
  for (int i = 0; i < natom; ++i) {
    for (int j = 0; j < natom; ++j) {
      if (i == j) continue;
      const double dx = atom_coords[3 * i + 0] - atom_coords[3 * j + 0];
      const double dy = atom_coords[3 * i + 1] - atom_coords[3 * j + 1];
      const double dz = atom_coords[3 * i + 2] - atom_coords[3 * j + 2];
      const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
      inv_rij[i * natom + j] = (d > 1e-12) ? (1.0 / d) : 0.0;
      if (radii_adjust) {
        const double chi = bragg_radii[i] / bragg_radii[j];
        const double u = (chi - 1.0) / (chi + 1.0);
        const double a = u / (u * u - 1.0);
        a_ij[i * natom + j] = std::clamp(a, -0.5, 0.5);
      }
    }
  }

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> dist(static_cast<std::size_t>(natom));
    std::vector<double> cell_p(static_cast<std::size_t>(natom));

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (std::ptrdiff_t ig = 0; ig < static_cast<std::ptrdiff_t>(n_raw); ++ig) {
      const double gx = raw_coords[3 * ig + 0];
      const double gy = raw_coords[3 * ig + 1];
      const double gz = raw_coords[3 * ig + 2];
      const int ia = parent_atom[static_cast<std::size_t>(ig)];

      for (int b = 0; b < natom; ++b) {
        const double dx = gx - atom_coords[3 * b + 0];
        const double dy = gy - atom_coords[3 * b + 1];
        const double dz = gz - atom_coords[3 * b + 2];
        dist[b] = std::sqrt(dx * dx + dy * dy + dz * dz);
      }

      double sum_p = 0.0;
      for (int b = 0; b < natom; ++b) {
        double pb = 1.0;
        for (int c = 0; c < natom; ++c) {
          if (b == c) continue;
          const double mu = (dist[b] - dist[c]) * inv_rij[b * natom + c];
          const double nu = mu + a_ij[b * natom + c] * (1.0 - mu * mu);
          const double s = (scheme == PartitionScheme::Stratmann) ? step_ssf(nu) : step_becke(nu);
          pb *= s;
          if (pb == 0.0) break;
        }
        cell_p[b] = pb;
        sum_p += pb;
      }

      if (sum_p > 0.0) {
        raw_weights[static_cast<std::size_t>(ig)] *= (cell_p[ia] / sum_p);
      } else {
        raw_weights[static_cast<std::size_t>(ig)] = 0.0;
      }
    }
  }
}

void build_octree_leaves(const std::vector<double>& coords,
                         std::vector<std::size_t>& indices,
                         std::size_t begin,
                         std::size_t end,
                         std::size_t max_leaf,
                         std::vector<std::pair<std::size_t, std::size_t>>& leaves) {
  const std::size_t count = end - begin;
  if (count == 0) return;
  if (count <= max_leaf) {
    leaves.emplace_back(begin, end);
    return;
  }

  // Compute bounding box of [begin, end)
  std::array<double, 3> bmin = {
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
  };
  std::array<double, 3> bmax = {
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };
  for (std::size_t k = begin; k < end; ++k) {
    const std::size_t idx = indices[k];
    for (int d = 0; d < 3; ++d) {
      const double v = coords[3 * idx + d];
      bmin[d] = std::min(bmin[d], v);
      bmax[d] = std::max(bmax[d], v);
    }
  }

  const double ex = bmax[0] - bmin[0];
  const double ey = bmax[1] - bmin[1];
  const double ez = bmax[2] - bmin[2];
  if (std::max({ex, ey, ez}) < 1e-12) {
    for (std::size_t pos = begin; pos < end; pos += max_leaf) {
      leaves.emplace_back(pos, std::min(pos + max_leaf, end));
    }
    return;
  }

  // 8-octant octree classification around bounding-box midpoint (or median fallback if degenerate)
  const std::array<double, 3> mid = {
      0.5 * (bmin[0] + bmax[0]),
      0.5 * (bmin[1] + bmax[1]),
      0.5 * (bmin[2] + bmax[2]),
  };

  std::array<std::vector<std::size_t>, 8> octants;
  for (std::size_t k = begin; k < end; ++k) {
    const std::size_t idx = indices[k];
    int code = 0;
    if (coords[3 * idx + 0] >= mid[0]) code |= 1;
    if (coords[3 * idx + 1] >= mid[1]) code |= 2;
    if (coords[3 * idx + 2] >= mid[2]) code |= 4;
    octants[code].push_back(idx);
  }

  int non_empty = 0;
  for (int o = 0; o < 8; ++o) {
    if (!octants[o].empty()) ++non_empty;
  }

  if (non_empty <= 1) {
    // Fallback: median split along longest axis
    int axis = 0;
    if (ey > ex && ey >= ez) axis = 1;
    else if (ez > ex && ez > ey) axis = 2;
    const std::size_t mid_idx = begin + count / 2;
    std::nth_element(
        indices.begin() + static_cast<std::ptrdiff_t>(begin),
        indices.begin() + static_cast<std::ptrdiff_t>(mid_idx),
        indices.begin() + static_cast<std::ptrdiff_t>(end),
        [&](std::size_t a, std::size_t b) {
          return coords[3 * a + axis] < coords[3 * b + axis];
        });
    build_octree_leaves(coords, indices, begin, mid_idx, max_leaf, leaves);
    build_octree_leaves(coords, indices, mid_idx, end, max_leaf, leaves);
    return;
  }

  // Write octants back in Z-order (Morton order!) and merge adjacent tiny sibling octants
  // when their combined size <= max_leaf so batches remain compact AND well-filled!
  std::size_t cursor = begin;
  std::array<std::pair<std::size_t, std::size_t>, 8> oct_ranges{};
  int n_ranges = 0;
  for (int o = 0; o < 8; ++o) {
    if (octants[o].empty()) continue;
    const std::size_t start = cursor;
    for (std::size_t idx : octants[o]) {
      indices[cursor++] = idx;
    }
    if (n_ranges > 0 &&
        (oct_ranges[n_ranges - 1].second - oct_ranges[n_ranges - 1].first) +
                (cursor - start) <=
            max_leaf) {
      oct_ranges[n_ranges - 1].second = cursor;
    } else {
      oct_ranges[n_ranges++] = {start, cursor};
    }
  }

  for (int r = 0; r < n_ranges; ++r) {
    build_octree_leaves(coords, indices, oct_ranges[r].first, oct_ranges[r].second,
                        max_leaf, leaves);
  }
}

}  // namespace

std::pair<std::vector<double>, std::vector<double>> MolecularGrid::lebedev_grid(int n_ang) {
  std::span<const OhOrbit> table;
  switch (n_ang) {
    case 14:  table = kLebedev14;  break;
    case 26:  table = kLebedev26;  break;
    case 50:  table = kLebedev50;  break;
    case 110: table = kLebedev110; break;
    case 194: table = kLebedev194; break;
    case 302: table = kLebedev302; break;
    default:
      throw std::invalid_argument(
          "Unsupported Lebedev grid size " + std::to_string(n_ang) +
          "; choose from {14, 26, 50, 110, 194, 302}");
  }

  std::vector<double> xyz;
  std::vector<double> weights;
  xyz.reserve(static_cast<std::size_t>(n_ang) * 3);
  weights.reserve(static_cast<std::size_t>(n_ang));
  for (const OhOrbit& orb : table) {
    append_oh_orbit(orb, xyz, weights);
  }
  return {std::move(xyz), std::move(weights)};
}

std::vector<double> MolecularGrid::compute_shell_radii(const Basis& basis, double ao_cutoff) {
  const double cutoff = std::max(ao_cutoff, 1e-18);
  std::vector<double> radii(basis.nshell(), 0.0);

  for (std::size_t is = 0; is < basis.nshell(); ++is) {
    const Shell& sh = basis.shell(is);
    const int l = sh.l;
    const double ang_bound = std::sqrt((2.0 * l + 1.0) / (4.0 * kPi));

    double min_alpha = sh.exponents[0];
    for (std::size_t k = 0; k < sh.exponents.size(); ++k) {
      min_alpha = std::min(min_alpha, sh.exponents[k]);
    }

    auto env_val = [&](double r) {
      double s = 0.0;
      const double rl = (l == 0) ? 1.0 : std::pow(r, l);
      for (std::size_t k = 0; k < sh.exponents.size(); ++k) {
        s += std::abs(sh.coefficients[k]) * std::exp(-sh.exponents[k] * r * r);
      }
      return ang_bound * rl * s;
    };

    const double r_peak = (l > 0) ? std::sqrt(static_cast<double>(l) / (2.0 * min_alpha)) : 0.0;
    double r_hi = std::max(r_peak + 1.0, std::sqrt(std::max(1.0, -std::log(cutoff)) / min_alpha));
    while (env_val(r_hi) > cutoff && r_hi < 1e4) {
      r_hi *= 1.5;
    }

    double r_lo = r_peak;
    for (int iter = 0; iter < 50; ++iter) {
      const double mid = 0.5 * (r_lo + r_hi);
      if (env_val(mid) > cutoff) {
        r_lo = mid;
      } else {
        r_hi = mid;
      }
    }
    radii[is] = r_hi;
  }
  return radii;
}

MolecularGrid::MolecularGrid(std::span<const XcAtom> atoms, const Basis& basis,
                             const GridOptions& opts)
    : nao_(basis.nao()) {
  if (atoms.empty()) {
    throw std::invalid_argument("MolecularGrid requires at least one atom");
  }
  if (opts.max_batch_points <= 0) {
    throw std::invalid_argument("GridOptions::max_batch_points must be positive");
  }

  // Cache angular tables for fast assembly
  static constexpr int kSupportedAng[] = {14, 26, 50, 110, 194, 302};
  std::array<std::pair<std::vector<double>, std::vector<double>>, 6> ang_cache;
  for (int i = 0; i < 6; ++i) {
    ang_cache[i] = lebedev_grid(kSupportedAng[i]);
  }
  auto get_ang = [&](int n_ang) -> const std::pair<std::vector<double>, std::vector<double>>& {
    for (int i = 0; i < 6; ++i) {
      if (kSupportedAng[i] == n_ang) return ang_cache[i];
    }
    throw std::invalid_argument("Unsupported Lebedev n_ang: " + std::to_string(n_ang));
  };

  if (opts.n_ang > 0) {
    (void)get_ang(opts.n_ang);  // Validate upfront
  }

  // Generate unpartitioned atom-centered Treutler-Ahlrichs M4 + Lebedev grids
  std::vector<double> raw_coords;
  std::vector<double> raw_weights;
  std::vector<int> parent_atom;

  const int max_ang = (opts.n_ang > 0) ? opts.n_ang : default_max_ang(opts.level);
  constexpr double kAlphaM4 = 0.6;
  const double inv_ln2 = 1.0 / std::log(2.0);

  for (int ia = 0; ia < static_cast<int>(atoms.size()); ++ia) {
    const int z = atoms[ia].atomic_number;
    const auto& ac = atoms[ia].center;
    const int n_rad = (opts.n_rad > 0) ? opts.n_rad : default_n_rad(z, opts.level);
    const double xi = treutler_radius_bohr(z);
    const double dtheta = kPi / static_cast<double>(n_rad + 1);

    for (int ir = 1; ir <= n_rad; ++ir) {
      const double theta = static_cast<double>(ir) * dtheta;
      const double x = std::cos(theta);
      const double sin_t = std::sin(theta);
      const double one_plus_x = 1.0 + x;
      const double one_minus_x = 1.0 - x;
      const double ln_term = std::log(2.0 / one_minus_x);
      const double pow_term = std::pow(one_plus_x, kAlphaM4);

      const double r = xi * inv_ln2 * pow_term * ln_term;
      const double dr_dtheta =
          xi * inv_ln2 * pow_term *
          ((kAlphaM4 * sin_t / one_plus_x) * ln_term + sin_t / one_minus_x);
      const double w_rad = r * r * dr_dtheta * dtheta;

      const int n_ang = (opts.n_ang > 0 && !opts.prune)
                            ? opts.n_ang
                            : pruned_lebedev_size(r, xi, max_ang);
      const auto& [uxyz, uw] = get_ang(n_ang);

      for (int j = 0; j < n_ang; ++j) {
        const double w_pt = w_rad * uw[j];
        if (w_pt <= 0.0) continue;
        raw_coords.push_back(ac[0] + r * uxyz[3 * j + 0]);
        raw_coords.push_back(ac[1] + r * uxyz[3 * j + 1]);
        raw_coords.push_back(ac[2] + r * uxyz[3 * j + 2]);
        raw_weights.push_back(w_pt);
        parent_atom.push_back(ia);
      }
    }
  }

  // Atomic Becke / Stratmann-Scuseria-Frisch partitioning (GPU or Host OpenMP)
  if (atoms.size() > 1 && !raw_weights.empty()) {
    std::vector<double> atom_coords(atoms.size() * 3);
    std::vector<double> bragg_radii(atoms.size());
    for (std::size_t ia = 0; ia < atoms.size(); ++ia) {
      atom_coords[3 * ia + 0] = atoms[ia].center[0];
      atom_coords[3 * ia + 1] = atoms[ia].center[1];
      atom_coords[3 * ia + 2] = atoms[ia].center[2];
      bragg_radii[ia] = bragg_slater_radius(atoms[ia].atomic_number);
    }

    bool ran_on_gpu = false;
#if defined(GPU4DFT_HAVE_CUDA)
    if (opts.backend == Backend::Cuda && detail::xc_cuda_device_count() > 0) {
      detail::compute_partition_weights_cuda(
          raw_coords.data(), parent_atom.data(), raw_weights.size(),
          atom_coords.data(), bragg_radii.data(), static_cast<int>(atoms.size()),
          opts.partition, opts.becke_radii_adjust, raw_weights.data());
      ran_on_gpu = true;
    }
#endif
    if (!ran_on_gpu) {
      compute_partition_weights_host(raw_coords, parent_atom, atom_coords, bragg_radii,
                                     opts.partition, opts.becke_radii_adjust, raw_weights);
    }
  }

  // Prune zero / negligible weight points
  std::vector<double> kept_coords;
  std::vector<double> kept_weights;
  kept_coords.reserve(raw_coords.size());
  kept_weights.reserve(raw_weights.size());
  for (std::size_t i = 0; i < raw_weights.size(); ++i) {
    if (std::abs(raw_weights[i]) >= opts.weight_cutoff) {
      kept_coords.push_back(raw_coords[3 * i + 0]);
      kept_coords.push_back(raw_coords[3 * i + 1]);
      kept_coords.push_back(raw_coords[3 * i + 2]);
      kept_weights.push_back(raw_weights[i]);
    }
  }

  finalize_spatial_batches(std::move(kept_coords), std::move(kept_weights),
                           basis, opts.max_batch_points, opts.ao_cutoff);
}

MolecularGrid MolecularGrid::from_points(std::span<const double> coords_xyz,
                                         std::span<const double> weights,
                                         const Basis& basis,
                                         int max_batch_points,
                                         double ao_cutoff) {
  if (coords_xyz.size() != weights.size() * 3) {
    throw std::invalid_argument(
        "MolecularGrid::from_points: coords_xyz.size() must equal 3 * weights.size()");
  }
  if (max_batch_points <= 0) {
    throw std::invalid_argument("MolecularGrid::from_points: max_batch_points must be positive");
  }

  MolecularGrid grid;
  grid.nao_ = basis.nao();
  std::vector<double> raw_c(coords_xyz.begin(), coords_xyz.end());
  std::vector<double> raw_w(weights.begin(), weights.end());
  grid.finalize_spatial_batches(std::move(raw_c), std::move(raw_w), basis,
                                max_batch_points, ao_cutoff);
  return grid;
}

void MolecularGrid::finalize_spatial_batches(std::vector<double> raw_coords,
                                             std::vector<double> raw_weights,
                                             const Basis& basis,
                                             int max_batch_points,
                                             double ao_cutoff) {
  const std::size_t npts = raw_weights.size();
  coords_.resize(npts * 3);
  weights_.resize(npts);
  batches_.clear();
  if (npts == 0) return;

  std::vector<std::size_t> order(npts);
  std::iota(order.begin(), order.end(), std::size_t{0});

  std::vector<std::pair<std::size_t, std::size_t>> leaves;
  build_octree_leaves(raw_coords, order, 0, npts,
                      static_cast<std::size_t>(max_batch_points), leaves);

  // Reorder coords & weights contiguously per spatial octree leaf
  for (std::size_t new_i = 0; new_i < npts; ++new_i) {
    const std::size_t old_i = order[new_i];
    coords_[3 * new_i + 0] = raw_coords[3 * old_i + 0];
    coords_[3 * new_i + 1] = raw_coords[3 * old_i + 1];
    coords_[3 * new_i + 2] = raw_coords[3 * old_i + 2];
    weights_[new_i] = raw_weights[old_i];
  }

  const std::vector<double> shell_radii = compute_shell_radii(basis, ao_cutoff);
  batches_.reserve(leaves.size());

  for (const auto& [beg, end] : leaves) {
    if (beg >= end) continue;
    SpatialBatch batch;
    batch.point_offset = beg;
    batch.npoints = end - beg;
    batch.box_min = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    batch.box_max = {
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (std::size_t p = beg; p < end; ++p) {
      for (int d = 0; d < 3; ++d) {
        const double v = coords_[3 * p + d];
        batch.box_min[d] = std::min(batch.box_min[d], v);
        batch.box_max[d] = std::max(batch.box_max[d], v);
      }
    }

    for (std::size_t is = 0; is < basis.nshell(); ++is) {
      const auto& sc = basis.shell(is).center;
      double d2 = 0.0;
      for (int d = 0; d < 3; ++d) {
        double dist_d = 0.0;
        if (sc[d] < batch.box_min[d]) dist_d = batch.box_min[d] - sc[d];
        else if (sc[d] > batch.box_max[d]) dist_d = sc[d] - batch.box_max[d];
        d2 += dist_d * dist_d;
      }
      const double rad = shell_radii[is];
      if (d2 <= rad * rad) {
        batch.active_shells.push_back(static_cast<int>(is));
        const int ao0 = basis.ao_offset(is);
        const int n_m = 2 * basis.shell(is).l + 1;
        for (int m = 0; m < n_m; ++m) {
          batch.active_aos.push_back(ao0 + m);
        }
      }
    }

    batches_.push_back(std::move(batch));
  }
}

}  // namespace gpu4dft
