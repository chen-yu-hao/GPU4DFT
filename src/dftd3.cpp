#include "gpu4dft/dftd3.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dftd3_tables.hpp"

namespace gpu4dft::dftd3 {
namespace {

constexpr double kK1 = 16.0;
constexpr double kK3 = 4.0;
constexpr double kCnCutoff = 40.0;     // Bohr (matches s-dftd3 & tad-dftd3)
constexpr double kDisp2Cutoff = 60.0;  // Bohr (matches s-dftd3 & tad-dftd3)

struct ParamEntry {
  const char* key;
  D3BJParams params;
};

constexpr ParamEntry kParamTable[] = {
    {"b3lyp", {1.0, 0.3981, 1.9889, 4.4211}},
    {"b3lyp5", {1.0, 0.3981, 1.9889, 4.4211}},
    {"b3lypg", {1.0, 0.3981, 1.9889, 4.4211}},
    {"pbe", {1.0, 0.4289, 0.7875, 4.4407}},
    {"pbe0", {1.0, 0.4145, 1.2177, 4.8593}},
    {"pbeh", {1.0, 0.4145, 1.2177, 4.8593}},
    {"camb3lyp", {1.0, 0.3708, 2.0674, 5.4743}},
    {"wb97x", {1.0, 0.0000, 0.2641, 5.4959}},
    {"hf", {1.0, 0.3385, 0.9171, 2.8830}},
    {"rhf", {1.0, 0.3385, 0.9171, 2.8830}},
    {"uhf", {1.0, 0.3385, 0.9171, 2.8830}},
    {"blyp", {1.0, 0.4298, 2.6996, 4.2359}},
    {"bp", {1.0, 0.3946, 3.2822, 4.8516}},
    {"bp86", {1.0, 0.3946, 3.2822, 4.8516}},
    {"tpss", {1.0, 0.4535, 1.9435, 4.4752}},
    {"tpssh", {1.0, 0.4529, 2.2382, 4.6550}},
    {"tpss0", {1.0, 0.3768, 1.2576, 4.5865}},
    {"revpbe", {1.0, 0.5238, 2.3550, 3.5016}},
    {"revpbe0", {1.0, 0.4679, 1.7588, 3.7619}},
    {"rpbe", {1.0, 0.1820, 0.8318, 4.0094}},
    {"pbesol", {1.0, 0.4466, 2.9491, 6.1742}},
    {"scan", {1.0, 0.5380, 0.0000, 5.4200}},
    {"r2scan", {1.0, 0.49484001, 0.78981345, 5.73083694}},
    {"rscan", {1.0, 0.47023427, 1.08859014, 5.73408312}},
    {"hse06", {1.0, 0.3830, 2.3100, 5.6850}},
    {"bhlyp", {1.0, 0.2793, 1.0354, 4.9615}},
    {"bhandhlyp", {1.0, 0.2793, 1.0354, 4.9615}},
    {"b3pw91", {1.0, 0.4312, 2.8524, 4.4693}},
    {"b97d", {1.0, 0.5545, 2.2609, 3.2297}},
    {"olyp", {1.0, 0.5299, 2.6205, 2.8065}},
    {"opbe", {1.0, 0.5512, 3.3816, 2.9444}},
    {"wb97m", {1.0, 0.5660, 0.3908, 3.1280}},
    {"lcwpbe", {1.0, 0.3919, 1.8541, 5.0897}},
    {"lcwhpbe", {1.0, 0.2746, 1.1908, 5.3157}},
    {"lcomegahpbe", {1.0, 0.2746, 1.1908, 5.3157}},
};

std::string normalize_key(std::string_view raw) {
  std::string out;
  out.reserve(raw.size());
  for (char ch : raw) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isspace(uch) != 0 || ch == '-' || ch == '_' || ch == '(' || ch == ')' || ch == '/') {
      continue;
    }
    out.push_back(static_cast<char>(std::tolower(uch)));
  }
  return out;
}

const D3BJParams* find_params(std::string_view functional) noexcept {
  const std::string norm = normalize_key(functional);
  for (const auto& e : kParamTable) {
    if (norm == e.key) return &e.params;
  }
  if (norm.size() > 4 && norm.compare(norm.size() - 4, 4, "d3bj") == 0) {
    const std::string base = norm.substr(0, norm.size() - 4);
    for (const auto& e : kParamTable) {
      if (base == e.key) return &e.params;
    }
  } else if (norm.size() > 2 && norm.compare(norm.size() - 2, 2, "d3") == 0) {
    const std::string base = norm.substr(0, norm.size() - 2);
    for (const auto& e : kParamTable) {
      if (base == e.key) return &e.params;
    }
  }
  return nullptr;
}

inline double lookup_c6_ref(int za, int ia, int zb, int jb) noexcept {
  const int a = detail::kRefOffset[za] + ia;
  const int b = detail::kRefOffset[zb] + jb;
  const int idx = (a <= b) ? (b * (b + 1) / 2 + a) : (a * (a + 1) / 2 + b);
  return detail::kC6UpperTri[idx];
}

}  // namespace

D3BJParams get_d3bj_params(std::string_view functional) {
  if (const D3BJParams* p = find_params(functional)) {
    return *p;
  }
  throw std::invalid_argument("Unsupported DFT-D3(BJ) functional: '" + std::string(functional) + "'");
}

bool has_d3bj_params(std::string_view functional) noexcept {
  return find_params(functional) != nullptr;
}

std::vector<std::string> supported_d3bj_functionals() {
  std::vector<std::string> out;
  out.reserve(sizeof(kParamTable) / sizeof(kParamTable[0]));
  for (const auto& e : kParamTable) {
    out.emplace_back(e.key);
  }
  return out;
}

D3BJResult compute_d3bj_with_params(const std::vector<int>& atomic_numbers,
                                    const std::vector<double>& coords_bohr,
                                    const D3BJParams& params,
                                    bool compute_grad) {
  const std::size_t natom = atomic_numbers.size();
  if (coords_bohr.size() != natom * 3) {
    throw std::invalid_argument(
        "compute_d3bj: coords_bohr size must equal 3 * atomic_numbers.size()");
  }
  for (std::size_t i = 0; i < natom; ++i) {
    const int z = atomic_numbers[i];
    if (z < 1 || z > detail::kMaxAtomicNumber) {
      throw std::invalid_argument(
          "compute_d3bj: atomic number out of supported range [1, 103]: " + std::to_string(z));
    }
  }

  D3BJResult res;
  res.cn.assign(natom, 0.0);
  if (compute_grad) {
    res.gradient.assign(natom * 3, 0.0);
  }
  if (natom <= 1) {
    return res;
  }

  const double cn_cut2 = kCnCutoff * kCnCutoff;
  const double disp_cut2 = kDisp2Cutoff * kDisp2Cutoff;

  // 1. Coordination numbers CN_A
  for (std::size_t i = 0; i < natom; ++i) {
    const double rcov_i = detail::kRcovBohr[atomic_numbers[i]];
    const double xi = coords_bohr[3 * i + 0];
    const double yi = coords_bohr[3 * i + 1];
    const double zi = coords_bohr[3 * i + 2];
    for (std::size_t j = i + 1; j < natom; ++j) {
      const double dx = xi - coords_bohr[3 * j + 0];
      const double dy = yi - coords_bohr[3 * j + 1];
      const double dz = zi - coords_bohr[3 * j + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < 1e-24) {
        throw std::invalid_argument("compute_d3bj: coincident atoms detected");
      }
      if (r2 <= cn_cut2) {
        const double r = std::sqrt(r2);
        const double rcov_ij = rcov_i + detail::kRcovBohr[atomic_numbers[j]];
        const double arg = -kK1 * (rcov_ij / r - 1.0);
        const double c_ij = 1.0 / (1.0 + std::exp(arg));
        res.cn[i] += c_ij;
        res.cn[j] += c_ij;
      }
    }
  }

  // 2. Gaussian reference weights w_A^i and derivatives dw_A^i / dCN_A
  std::vector<std::array<double, 7>> weights(natom);
  std::vector<std::array<double, 7>> dweights(natom);
  for (std::size_t i = 0; i < natom; ++i) {
    const int z = atomic_numbers[i];
    const int nref = detail::kRefCount[z];
    const double cn = res.cn[i];
    weights[i].fill(0.0);
    dweights[i].fill(0.0);

    double sum_l = 0.0;
    double sum_dl = 0.0;
    std::array<double, 7> l_vals{};
    std::array<double, 7> dl_vals{};
    int max_ref_idx = 0;
    double max_cn_ref = detail::kCnRef[z][0];

    for (int r = 0; r < nref; ++r) {
      const double cn_ref = detail::kCnRef[z][r];
      if (cn_ref > max_cn_ref) {
        max_cn_ref = cn_ref;
        max_ref_idx = r;
      }
      const double diff = cn - cn_ref;
      const double lv = std::exp(-kK3 * diff * diff);
      const double dlv = -2.0 * kK3 * diff * lv;
      l_vals[r] = lv;
      dl_vals[r] = dlv;
      sum_l += lv;
      sum_dl += dlv;
    }

    if (sum_l < 1e-300) {
      weights[i][max_ref_idx] = 1.0;
    } else {
      const double inv_sum = 1.0 / sum_l;
      for (int r = 0; r < nref; ++r) {
        const double w = l_vals[r] * inv_sum;
        weights[i][r] = w;
        if (compute_grad) {
          dweights[i][r] = (dl_vals[r] - w * sum_dl) * inv_sum;
        }
      }
    }
  }

  // 3. Two-body D3(BJ) energy, direct pair gradients, and dE/dCN adjoints
  std::vector<double> de_dcn(compute_grad ? natom : 0, 0.0);
  double total_energy = 0.0;

  for (std::size_t i = 0; i < natom; ++i) {
    const int zi = atomic_numbers[i];
    const int nref_i = detail::kRefCount[zi];
    const double r4r2_i = detail::kR4R2[zi];
    const double xi = coords_bohr[3 * i + 0];
    const double yi = coords_bohr[3 * i + 1];
    const double zi_c = coords_bohr[3 * i + 2];

    for (std::size_t j = i + 1; j < natom; ++j) {
      const double dx = xi - coords_bohr[3 * j + 0];
      const double dy = yi - coords_bohr[3 * j + 1];
      const double dz = zi_c - coords_bohr[3 * j + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > disp_cut2) continue;

      const int zj = atomic_numbers[j];
      const int nref_j = detail::kRefCount[zj];

      double c6 = 0.0;
      double dc6_dcni = 0.0;
      double dc6_dcnj = 0.0;
      for (int a = 0; a < nref_i; ++a) {
        const double wi = weights[i][a];
        const double dwi = dweights[i][a];
        double row_w = 0.0;
        double row_dw = 0.0;
        for (int b = 0; b < nref_j; ++b) {
          const double c6_ab = lookup_c6_ref(zi, a, zj, b);
          row_w += weights[j][b] * c6_ab;
          if (compute_grad) {
            row_dw += dweights[j][b] * c6_ab;
          }
        }
        c6 += wi * row_w;
        if (compute_grad) {
          dc6_dcni += dwi * row_w;
          dc6_dcnj += wi * row_dw;
        }
      }

      const double q_ij = 3.0 * r4r2_i * detail::kR4R2[zj];
      const double r0_ij = std::sqrt(q_ij);
      const double rbj = params.a1 * r0_ij + params.a2;
      const double rbj2 = rbj * rbj;
      const double rbj6 = rbj2 * rbj2 * rbj2;
      const double rbj8 = rbj6 * rbj2;

      const double r4 = r2 * r2;
      const double r6 = r4 * r2;
      const double r8 = r4 * r4;

      const double t6 = 1.0 / (r6 + rbj6);
      const double t8 = 1.0 / (r8 + rbj8);
      const double phi = params.s6 * t6 + params.s8 * q_ij * t8;

      total_energy -= c6 * phi;

      if (compute_grad) {
        // Direct derivative w.r.t. R_ij: (1/R_ij) * dE_ij/dR_ij
        // dphi/dR = -6*s6*R^5*t6^2 - 8*s8*q*R^7*t8^2
        // -(C6 / R) * dphi/dR = C6 * (6*s6*R^4*t6^2 + 8*s8*q*R^6*t8^2)
        const double d_dir = c6 * (6.0 * params.s6 * r4 * t6 * t6 +
                                   8.0 * params.s8 * q_ij * r6 * t8 * t8);
        res.gradient[3 * i + 0] += d_dir * dx;
        res.gradient[3 * i + 1] += d_dir * dy;
        res.gradient[3 * i + 2] += d_dir * dz;
        res.gradient[3 * j + 0] -= d_dir * dx;
        res.gradient[3 * j + 1] -= d_dir * dy;
        res.gradient[3 * j + 2] -= d_dir * dz;

        de_dcn[i] -= dc6_dcni * phi;
        de_dcn[j] -= dc6_dcnj * phi;
      }
    }
  }

  // 4. Indirect CN chain-rule gradient pass (pairs within cn_cutoff)
  if (compute_grad) {
    for (std::size_t i = 0; i < natom; ++i) {
      const double rcov_i = detail::kRcovBohr[atomic_numbers[i]];
      const double xi = coords_bohr[3 * i + 0];
      const double yi = coords_bohr[3 * i + 1];
      const double zi_c = coords_bohr[3 * i + 2];
      for (std::size_t j = i + 1; j < natom; ++j) {
        const double dx = xi - coords_bohr[3 * j + 0];
        const double dy = yi - coords_bohr[3 * j + 1];
        const double dz = zi_c - coords_bohr[3 * j + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > cn_cut2) continue;

        const double r = std::sqrt(r2);
        const double rcov_ij = rcov_i + detail::kRcovBohr[atomic_numbers[j]];
        const double arg = -kK1 * (rcov_ij / r - 1.0);
        const double c_ij = 1.0 / (1.0 + std::exp(arg));
        // (1/r) * dc_ij/dr = -k1 * rcov_ij * c_ij * (1 - c_ij) / r^3
        const double dc_over_r = -kK1 * rcov_ij * c_ij * (1.0 - c_ij) / (r2 * r);
        const double factor = (de_dcn[i] + de_dcn[j]) * dc_over_r;

        res.gradient[3 * i + 0] += factor * dx;
        res.gradient[3 * i + 1] += factor * dy;
        res.gradient[3 * i + 2] += factor * dz;
        res.gradient[3 * j + 0] -= factor * dx;
        res.gradient[3 * j + 1] -= factor * dy;
        res.gradient[3 * j + 2] -= factor * dz;
      }
    }
  }

  res.energy = total_energy;
  return res;
}

D3BJResult compute_d3bj(const std::vector<int>& atomic_numbers,
                        const std::vector<double>& coords_bohr,
                        const std::string& functional,
                        bool compute_grad) {
  return compute_d3bj_with_params(atomic_numbers, coords_bohr, get_d3bj_params(functional),
                                  compute_grad);
}

D3BJResult compute_d3bj(std::span<const Atom> atoms,
                        std::string_view functional,
                        bool compute_grad) {
  std::vector<int> atomic_numbers;
  std::vector<double> coords_bohr;
  atomic_numbers.reserve(atoms.size());
  coords_bohr.reserve(atoms.size() * 3);
  for (const Atom& a : atoms) {
    atomic_numbers.push_back(a.atomic_number);
    coords_bohr.push_back(a.center[0]);
    coords_bohr.push_back(a.center[1]);
    coords_bohr.push_back(a.center[2]);
  }
  return compute_d3bj_with_params(atomic_numbers, coords_bohr, get_d3bj_params(functional),
                                  compute_grad);
}

}  // namespace gpu4dft::dftd3
