#include "gpu4dft/one_electron.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/eri.hpp"

namespace gpu4dft {
namespace {

constexpr int kMaxSupportedL = 4;

struct HostBasisTable {
  std::vector<int> shell_l;
  std::vector<int> shell_ao;
  std::vector<int> shell_atom;
  std::vector<int> prim_offset{0};
  std::vector<double> shell_center;
  std::vector<double> prim_exp;
  std::vector<double> prim_coeff;
  std::vector<int> atom_Z;
  std::vector<double> atom_center;
  int nao = 0;

  detail::BasisDeviceView view() const {
    return detail::BasisDeviceView{
        static_cast<int>(shell_l.size()),
        nao,
        static_cast<int>(atom_Z.size()),
        shell_l.data(),
        shell_ao.data(),
        shell_atom.data(),
        prim_offset.data(),
        shell_center.data(),
        prim_exp.data(),
        prim_coeff.data(),
        atom_Z.data(),
        atom_center.data(),
    };
  }
};

HostBasisTable pack_basis_and_atoms(const Basis& basis, std::span<const Atom> atoms) {
  if (basis.max_l() > kMaxSupportedL) {
    throw NotImplemented("1e integrals support up to g shells (l <= 4); basis has l = " +
                         std::to_string(basis.max_l()));
  }
  HostBasisTable t;
  t.nao = basis.nao();
  const std::size_t nsh = basis.nshell();
  t.shell_l.reserve(nsh);
  t.shell_ao.reserve(nsh);
  t.shell_atom.reserve(nsh);
  t.shell_center.reserve(nsh * 3);
  t.prim_offset.reserve(nsh + 1);

  t.atom_Z.reserve(atoms.size());
  t.atom_center.reserve(atoms.size() * 3);
  for (const Atom& a : atoms) {
    t.atom_Z.push_back(a.atomic_number);
    t.atom_center.push_back(a.center[0]);
    t.atom_center.push_back(a.center[1]);
    t.atom_center.push_back(a.center[2]);
  }

  for (std::size_t i = 0; i < nsh; ++i) {
    const Shell& s = basis.shell(i);
    t.shell_l.push_back(s.l);
    t.shell_ao.push_back(basis.ao_offset(i));
    t.shell_center.push_back(s.center[0]);
    t.shell_center.push_back(s.center[1]);
    t.shell_center.push_back(s.center[2]);

    int matched_atom = -1;
    for (std::size_t ia = 0; ia < atoms.size(); ++ia) {
      const double dx = s.center[0] - atoms[ia].center[0];
      const double dy = s.center[1] - atoms[ia].center[1];
      const double dz = s.center[2] - atoms[ia].center[2];
      if (dx * dx + dy * dy + dz * dz < 1e-12) {
        matched_atom = static_cast<int>(ia);
        break;
      }
    }
    t.shell_atom.push_back(matched_atom);

    for (std::size_t k = 0; k < s.exponents.size(); ++k) {
      t.prim_exp.push_back(s.exponents[k]);
      t.prim_coeff.push_back(s.coefficients[k]);
    }
    t.prim_offset.push_back(static_cast<int>(t.prim_exp.size()));
  }
  return t;
}

std::vector<double> eval_one_elec_matrix(const Basis& basis, std::span<const Atom> atoms, int op,
                                         Backend backend) {
  if (!backend_available(backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(backend) + "' is not available");
  }
  const HostBasisTable table = pack_basis_and_atoms(basis, atoms);
  const std::size_t nao = static_cast<std::size_t>(basis.nao());
  std::vector<double> out(nao * nao, 0.0);
  if (nao == 0) return out;

  if (backend == Backend::Host) {
    const detail::BasisDeviceView v = table.view();
    const int nsh = v.nshell;
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(static_cast<std::size_t>(nsh) * (nsh + 1) / 2);
    for (int i = 0; i < nsh; ++i) {
      for (int j = 0; j <= i; ++j) pairs.emplace_back(i, j);
    }
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 16)
    #endif
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(pairs.size()); ++idx) {
      detail::eval_one_elec_pair(v, pairs[idx].first, pairs[idx].second, op, out.data());
    }
    return out;
  }

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::run_one_electron_matrix_cuda(table.view(), op, out);
    return out;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  throw std::runtime_error("gpu4dft was built without HIP");
}

}  // namespace

std::vector<double> overlap_matrix(const Basis& basis, Backend backend) {
  return eval_one_elec_matrix(basis, {}, 0, backend);
}

std::vector<double> kinetic_matrix(const Basis& basis, Backend backend) {
  return eval_one_elec_matrix(basis, {}, 1, backend);
}

std::vector<double> nuclear_attraction_matrix(const Basis& basis, std::span<const Atom> atoms,
                                              Backend backend) {
  return eval_one_elec_matrix(basis, atoms, 2, backend);
}

double nuclear_repulsion_energy(std::span<const Atom> atoms) {
  double energy = 0.0;
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    for (std::size_t j = i + 1; j < atoms.size(); ++j) {
      const double dx = atoms[i].center[0] - atoms[j].center[0];
      const double dy = atoms[i].center[1] - atoms[j].center[1];
      const double dz = atoms[i].center[2] - atoms[j].center[2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r < 1e-12) {
        throw std::invalid_argument("coincident nuclei in nuclear_repulsion_energy");
      }
      energy += static_cast<double>(atoms[i].atomic_number * atoms[j].atomic_number) / r;
    }
  }
  return energy;
}

std::vector<double> nuclear_repulsion_grad(std::span<const Atom> atoms) {
  std::vector<double> grad(atoms.size() * 3, 0.0);
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    for (std::size_t j = i + 1; j < atoms.size(); ++j) {
      const double dx = atoms[i].center[0] - atoms[j].center[0];
      const double dy = atoms[i].center[1] - atoms[j].center[1];
      const double dz = atoms[i].center[2] - atoms[j].center[2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      const double r = std::sqrt(r2);
      if (r < 1e-12) {
        throw std::invalid_argument("coincident nuclei in nuclear_repulsion_grad");
      }
      const double zz_over_r3 =
          static_cast<double>(atoms[i].atomic_number * atoms[j].atomic_number) / (r2 * r);
      const double gx = -zz_over_r3 * dx;
      const double gy = -zz_over_r3 * dy;
      const double gz = -zz_over_r3 * dz;
      grad[3 * i + 0] += gx;
      grad[3 * i + 1] += gy;
      grad[3 * i + 2] += gz;
      grad[3 * j + 0] -= gx;
      grad[3 * j + 1] -= gy;
      grad[3 * j + 2] -= gz;
    }
  }
  return grad;
}

std::vector<double> one_electron_grad(const Basis& basis, std::span<const Atom> atoms,
                                      const double* dm, const double* energy_weighted_dm,
                                      Backend backend) {
  if (!backend_available(backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(backend) + "' is not available");
  }
  std::vector<double> grad(atoms.size() * 3, 0.0);
  if (atoms.empty() || basis.nao() == 0) return grad;
  if (dm == nullptr || energy_weighted_dm == nullptr) {
    throw std::invalid_argument("one_electron_grad requires non-null dm and energy_weighted_dm");
  }

  const HostBasisTable table = pack_basis_and_atoms(basis, atoms);
  if (backend == Backend::Host) {
    const detail::BasisDeviceView v = table.view();
    const int nsh = v.nshell;
    const int total_pairs = nsh * nsh;

    #if defined(_OPENMP)
    #pragma omp parallel
    #endif
    {
      std::vector<double> local_grad(atoms.size() * 3, 0.0);
      auto add_hf = [&](int c, double fx, double fy, double fz) {
        local_grad[3 * c + 0] += fx;
        local_grad[3 * c + 1] += fy;
        local_grad[3 * c + 2] += fz;
      };

      #if defined(_OPENMP)
      #pragma omp for schedule(dynamic, 8)
      #endif
      for (int idx = 0; idx < total_pairs; ++idx) {
        const int ish = idx / nsh;
        const int jsh = idx - ish * nsh;
        double bra_f[3] = {0.0, 0.0, 0.0};
        detail::eval_one_elec_grad_ordered_pair(v, ish, jsh, dm, energy_weighted_dm, bra_f,
                                                add_hf);
        const int at_i = v.shell_atom[ish];
        if (at_i >= 0) {
          local_grad[3 * at_i + 0] += bra_f[0];
          local_grad[3 * at_i + 1] += bra_f[1];
          local_grad[3 * at_i + 2] += bra_f[2];
        }
      }

      #if defined(_OPENMP)
      #pragma omp critical
      #endif
      {
        for (std::size_t k = 0; k < grad.size(); ++k) grad[k] += local_grad[k];
      }
    }
    return grad;
  }

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::run_one_electron_grad_cuda(table.view(), dm, energy_weighted_dm, grad);
    return grad;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  throw std::runtime_error("gpu4dft was built without HIP");
}

}  // namespace gpu4dft
