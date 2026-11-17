#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/eri.hpp"
#include "gpu4dft/jk_engine.hpp"
#include "gpu4dft/kernel_registry.hpp"
#include "gpu4dft/linalg.hpp"
#include "gpu4dft/one_electron.hpp"
#include "gpu4dft/pw.hpp"
#include "gpu4dft/response.hpp"
#include "gpu4dft/scf.hpp"
#include "gpu4dft/snk.hpp"
#include "gpu4dft/types.hpp"
#include "gpu4dft/workflow.hpp"
#include "gpu4dft/xc.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using ConstVec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ConstMat = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using ConstComplex4D =
    nb::ndarray<const std::complex<double>, nb::ndim<4>, nb::c_contig, nb::device::cpu>;
using OutVec = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using OutMat = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using OutTen3 = nb::ndarray<nb::numpy, double, nb::ndim<3>>;
using OutTen4 = nb::ndarray<nb::numpy, double, nb::ndim<4>>;
using OutIntVec = nb::ndarray<nb::numpy, int, nb::ndim<1>>;
using OutComplexMat = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<2>>;
using OutComplex4D = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<4>>;

OutVec to_numpy_1d(std::vector<double> data, std::size_t n0) {
  auto* owned = new std::vector<double>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return OutVec(owned->data(), {n0}, owner);
}

OutMat to_numpy_2d(std::vector<double> data, std::size_t n0, std::size_t n1) {
  auto* owned = new std::vector<double>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return OutMat(owned->data(), {n0, n1}, owner);
}

OutTen3 to_numpy_3d(std::vector<double> data, std::size_t n0, std::size_t n1, std::size_t n2) {
  auto* owned = new std::vector<double>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return OutTen3(owned->data(), {n0, n1, n2}, owner);
}

OutTen4 to_numpy_4d(std::vector<double> data, std::size_t n) {
  auto* owned = new std::vector<double>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return OutTen4(owned->data(), {n, n, n, n}, owner);
}

OutIntVec to_numpy_int_1d(std::vector<int> data, std::size_t n0) {
  auto* owned = new std::vector<int>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<int>*>(p); });
  return OutIntVec(owned->data(), {n0}, owner);
}

OutComplexMat to_numpy_complex_2d(std::vector<std::complex<double>> data, std::size_t n0,
                                  std::size_t n1) {
  auto* owned = new std::vector<std::complex<double>>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept {
    delete static_cast<std::vector<std::complex<double>>*>(p);
  });
  return OutComplexMat(owned->data(), {n0, n1}, owner);
}

OutComplex4D to_numpy_complex_4d(std::vector<std::complex<double>> data, std::size_t n0,
                                 std::size_t n1, std::size_t n2, std::size_t n3) {
  auto* owned = new std::vector<std::complex<double>>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept {
    delete static_cast<std::vector<std::complex<double>>*>(p);
  });
  return OutComplex4D(owned->data(), {n0, n1, n2, n3}, owner);
}

std::array<std::array<double, 3>, 3> mat_to_lattice(const ConstMat& a) {
  if (a.shape(0) != 3 || a.shape(1) != 3) {
    throw std::invalid_argument("lattice vectors a_bohr must have shape (3, 3)");
  }
  const double* p = a.data();
  std::array<std::array<double, 3>, 3> out{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = p[3 * i + j];
    }
  }
  return out;
}

std::vector<gpu4dft::pw::PwAtom> make_pw_atoms(const std::vector<int>& atomic_numbers,
                                               const ConstMat& coords_bohr) {
  if (coords_bohr.shape(0) != atomic_numbers.size() || coords_bohr.shape(1) != 3) {
    throw std::invalid_argument(
        "coords_bohr must have shape (len(atomic_numbers), 3)");
  }
  const double* p = coords_bohr.data();
  std::vector<gpu4dft::pw::PwAtom> atoms(atomic_numbers.size());
  for (std::size_t i = 0; i < atomic_numbers.size(); ++i) {
    atoms[i].atomic_number = atomic_numbers[i];
    atoms[i].pos = {p[3 * i + 0], p[3 * i + 1], p[3 * i + 2]};
  }
  return atoms;
}

void check_square(const ConstMat& m, int nao, const char* name) {
  if (static_cast<int>(m.shape(0)) != nao || static_cast<int>(m.shape(1)) != nao) {
    throw std::invalid_argument(std::string(name) + " must have shape (" + std::to_string(nao) +
                                ", " + std::to_string(nao) + ")");
  }
}

std::vector<gpu4dft::XcAtom> atoms_to_xc(const std::vector<gpu4dft::Atom>& atoms) {
  std::vector<gpu4dft::XcAtom> out;
  out.reserve(atoms.size());
  for (const auto& a : atoms) {
    out.push_back(gpu4dft::XcAtom{a.atomic_number, a.center});
  }
  return out;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "GPU4DFT native C++20/CUDA core";

  nb::exception<gpu4dft::NotImplemented>(m, "NotImplementedError", PyExc_NotImplementedError);

  // -------------------------------------------------------------------------
  // Core Enums
  // -------------------------------------------------------------------------
  nb::enum_<gpu4dft::Backend>(m, "Backend")
      .value("host", gpu4dft::Backend::Host)
      .value("cuda", gpu4dft::Backend::Cuda)
      .value("hip", gpu4dft::Backend::Hip);

  nb::enum_<gpu4dft::Precision>(m, "Precision")
      .value("fp64", gpu4dft::Precision::FP64)
      .value("fp32", gpu4dft::Precision::FP32)
      .value("mixed", gpu4dft::Precision::Mixed);

  nb::enum_<gpu4dft::GemmMode>(m, "GemmMode")
      .value("fp64", gpu4dft::GemmMode::FP64)
      .value("fp32", gpu4dft::GemmMode::FP32)
      .value("ozaki_int8", gpu4dft::GemmMode::OzakiInt8);

  nb::enum_<gpu4dft::DensitySolver>(m, "DensitySolver")
      .value("diag", gpu4dft::DensitySolver::Diag)
      .value("sp2", gpu4dft::DensitySolver::SP2);

  nb::enum_<gpu4dft::PartitionScheme>(m, "PartitionScheme")
      .value("stratmann", gpu4dft::PartitionScheme::Stratmann)
      .value("becke", gpu4dft::PartitionScheme::Becke);

  nb::enum_<gpu4dft::JKMode>(m, "JKMode")
      .value("direct", gpu4dft::JKMode::Direct)
      .value("ri", gpu4dft::JKMode::RI)
      .value("snk", gpu4dft::JKMode::SnK)
      .value("ri_snk", gpu4dft::JKMode::RISnK);

  nb::enum_<gpu4dft::ResponseMethod>(m, "ResponseMethod")
      .value("tda", gpu4dft::ResponseMethod::TDA)
      .value("tddft", gpu4dft::ResponseMethod::TDDFT);

  // -------------------------------------------------------------------------
  // Basis, Shell & Atom
  // -------------------------------------------------------------------------
  nb::class_<gpu4dft::Shell>(m, "Shell")
      .def(
          "__init__",
          [](gpu4dft::Shell* self, int l, std::array<double, 3> center,
             std::vector<double> exponents, std::vector<double> coefficients) {
            new (self) gpu4dft::Shell{l, center, std::move(exponents), std::move(coefficients)};
          },
          "l"_a, "center"_a, "exponents"_a, "coefficients"_a)
      .def_rw("l", &gpu4dft::Shell::l)
      .def_rw("center", &gpu4dft::Shell::center)
      .def_rw("exponents", &gpu4dft::Shell::exponents)
      .def_rw("coefficients", &gpu4dft::Shell::coefficients);

  nb::class_<gpu4dft::Basis>(m, "Basis")
      .def(nb::init<std::vector<gpu4dft::Shell>>(), "shells"_a)
      .def_prop_ro("nao", &gpu4dft::Basis::nao)
      .def_prop_ro("nshell", &gpu4dft::Basis::nshell)
      .def_prop_ro("max_l", &gpu4dft::Basis::max_l)
      .def_prop_ro("shells", &gpu4dft::Basis::shells)
      .def("ao_offset", &gpu4dft::Basis::ao_offset, "ishell"_a);

  nb::class_<gpu4dft::Atom>(m, "Atom")
      .def(
          "__init__",
          [](gpu4dft::Atom* self, int atomic_number, std::array<double, 3> center) {
            new (self) gpu4dft::Atom{atomic_number, center};
          },
          "atomic_number"_a, "center"_a)
      .def_rw("atomic_number", &gpu4dft::Atom::atomic_number)
      .def_rw("center", &gpu4dft::Atom::center);

  m.def("backend_available", &gpu4dft::backend_available, "backend"_a);

  m.def(
      "eri_tensor",
      [](const gpu4dft::Basis& basis, gpu4dft::Backend backend, gpu4dft::Precision precision) {
        std::vector<double> eri;
        {
          nb::gil_scoped_release release;
          eri = gpu4dft::eri_tensor(basis, backend, precision);
        }
        return to_numpy_4d(std::move(eri), static_cast<std::size_t>(basis.nao()));
      },
      "basis"_a, "backend"_a, "precision"_a,
      "Full (ij|kl) ERI tensor, shape (nao, nao, nao, nao). Validation only.");

  m.def(
      "kernel_names",
      [] {
        std::vector<std::string> names;
        for (const gpu4dft::EriKernel& k : gpu4dft::eri_kernels()) names.emplace_back(k.name);
        return names;
      },
      "Names of the generated ERI kernels compiled into this build.");

  // -------------------------------------------------------------------------
  // One-Electron Integrals & Nuclear Gradients
  // -------------------------------------------------------------------------
  m.def(
      "overlap_matrix",
      [](const gpu4dft::Basis& basis, gpu4dft::Backend backend) {
        std::vector<double> mat;
        {
          nb::gil_scoped_release release;
          mat = gpu4dft::overlap_matrix(basis, backend);
        }
        const auto n = static_cast<std::size_t>(basis.nao());
        return to_numpy_2d(std::move(mat), n, n);
      },
      "basis"_a, "backend"_a = gpu4dft::Backend::Host);

  m.def(
      "kinetic_matrix",
      [](const gpu4dft::Basis& basis, gpu4dft::Backend backend) {
        std::vector<double> mat;
        {
          nb::gil_scoped_release release;
          mat = gpu4dft::kinetic_matrix(basis, backend);
        }
        const auto n = static_cast<std::size_t>(basis.nao());
        return to_numpy_2d(std::move(mat), n, n);
      },
      "basis"_a, "backend"_a = gpu4dft::Backend::Host);

  m.def(
      "nuclear_attraction_matrix",
      [](const gpu4dft::Basis& basis, const std::vector<gpu4dft::Atom>& atoms,
         gpu4dft::Backend backend) {
        std::vector<double> mat;
        {
          nb::gil_scoped_release release;
          mat = gpu4dft::nuclear_attraction_matrix(basis, atoms, backend);
        }
        const auto n = static_cast<std::size_t>(basis.nao());
        return to_numpy_2d(std::move(mat), n, n);
      },
      "basis"_a, "atoms"_a, "backend"_a = gpu4dft::Backend::Host);

  m.def(
      "nuclear_repulsion_energy",
      [](const std::vector<gpu4dft::Atom>& atoms) {
        return gpu4dft::nuclear_repulsion_energy(atoms);
      },
      "atoms"_a);

  m.def(
      "nuclear_repulsion_grad",
      [](const std::vector<gpu4dft::Atom>& atoms) {
        std::vector<double> g = gpu4dft::nuclear_repulsion_grad(atoms);
        return to_numpy_2d(std::move(g), atoms.size(), 3);
      },
      "atoms"_a);

  m.def(
      "one_electron_grad",
      [](const gpu4dft::Basis& basis, const std::vector<gpu4dft::Atom>& atoms, ConstMat dm,
         ConstMat wdm, gpu4dft::Backend backend) {
        check_square(dm, basis.nao(), "dm");
        check_square(wdm, basis.nao(), "wdm");
        const double* dm_ptr = dm.data();
        const double* wdm_ptr = wdm.data();
        std::vector<double> g;
        {
          nb::gil_scoped_release release;
          g = gpu4dft::one_electron_grad(basis, atoms, dm_ptr, wdm_ptr, backend);
        }
        return to_numpy_2d(std::move(g), atoms.size(), 3);
      },
      "basis"_a, "atoms"_a, "dm"_a, "wdm"_a, "backend"_a = gpu4dft::Backend::Cuda);

  // -------------------------------------------------------------------------
  // Direct J/K, RI-JK & 2e Nuclear Gradients
  // -------------------------------------------------------------------------
  nb::class_<gpu4dft::JKOptions>(m, "JKOptions")
      .def(nb::init<>())
      .def_rw("backend", &gpu4dft::JKOptions::backend)
      .def_rw("precision", &gpu4dft::JKOptions::precision)
      .def_rw("schwarz_cutoff", &gpu4dft::JKOptions::schwarz_cutoff)
      .def_rw("fp32_threshold", &gpu4dft::JKOptions::fp32_threshold)
      .def_rw("compute_j", &gpu4dft::JKOptions::compute_j)
      .def_rw("compute_k", &gpu4dft::JKOptions::compute_k)
      .def_rw("k_scale", &gpu4dft::JKOptions::k_scale)
      .def_rw("symmetrize_dm", &gpu4dft::JKOptions::symmetrize_dm);

  nb::class_<gpu4dft::DirectJKEngine>(m, "DirectJKEngine")
      .def(
          "__init__",
          [](gpu4dft::DirectJKEngine* self, const gpu4dft::Basis& basis, double prim_cutoff) {
            nb::gil_scoped_release release;
            new (self) gpu4dft::DirectJKEngine(basis, prim_cutoff);
          },
          "basis"_a, "prim_cutoff"_a = 1e-14)
      .def_prop_ro("nao", &gpu4dft::DirectJKEngine::nao)
      .def(
          "compute_jk",
          [](const gpu4dft::DirectJKEngine& self, ConstMat dm, const gpu4dft::JKOptions& opts) {
            const int n = self.nao();
            check_square(dm, n, "dm");
            const double* dm_ptr = dm.data();
            const auto nu = static_cast<std::size_t>(n);
            std::vector<double> vj(nu * nu, 0.0);
            std::vector<double> vk(nu * nu, 0.0);
            {
              nb::gil_scoped_release release;
              self.compute_jk(dm_ptr, opts.compute_j ? vj.data() : nullptr,
                              opts.compute_k ? vk.data() : nullptr, opts);
            }
            return std::make_pair(to_numpy_2d(std::move(vj), nu, nu),
                                  to_numpy_2d(std::move(vk), nu, nu));
          },
          "dm"_a, "opts"_a = gpu4dft::JKOptions{})
      .def(
          "compute_incremental_jk",
          [](const gpu4dft::DirectJKEngine& self, ConstMat dm_new, ConstMat dm_old,
             ConstMat vj_prev, ConstMat vk_prev, const gpu4dft::JKOptions& opts) {
            const int n = self.nao();
            check_square(dm_new, n, "dm_new");
            check_square(dm_old, n, "dm_old");
            check_square(vj_prev, n, "vj_prev");
            check_square(vk_prev, n, "vk_prev");
            const auto nu = static_cast<std::size_t>(n);
            std::vector<double> vj(vj_prev.data(), vj_prev.data() + nu * nu);
            std::vector<double> vk(vk_prev.data(), vk_prev.data() + nu * nu);
            const double* d_new = dm_new.data();
            const double* d_old = dm_old.data();
            {
              nb::gil_scoped_release release;
              self.compute_incremental_jk(d_new, d_old, opts.compute_j ? vj.data() : nullptr,
                                          opts.compute_k ? vk.data() : nullptr, opts);
            }
            return std::make_pair(to_numpy_2d(std::move(vj), nu, nu),
                                  to_numpy_2d(std::move(vk), nu, nu));
          },
          "dm_new"_a, "dm_old"_a, "vj_prev"_a, "vk_prev"_a, "opts"_a = gpu4dft::JKOptions{});

  nb::class_<gpu4dft::RIJKEngine>(m, "RIJKEngine")
      .def(
          "__init__",
          [](gpu4dft::RIJKEngine* self, const gpu4dft::Basis& basis,
             const gpu4dft::Basis& aux_basis, gpu4dft::Backend backend) {
            nb::gil_scoped_release release;
            new (self) gpu4dft::RIJKEngine(basis, aux_basis, backend);
          },
          "basis"_a, "aux_basis"_a, "backend"_a = gpu4dft::Backend::Cuda)
      .def_prop_ro("nao", &gpu4dft::RIJKEngine::nao)
      .def_prop_ro("naux", &gpu4dft::RIJKEngine::naux)
      .def(
          "compute_jk",
          [](const gpu4dft::RIJKEngine& self, ConstMat dm, const gpu4dft::JKOptions& opts) {
            const int n = self.nao();
            check_square(dm, n, "dm");
            const double* dm_ptr = dm.data();
            const auto nu = static_cast<std::size_t>(n);
            std::vector<double> vj(nu * nu, 0.0);
            std::vector<double> vk(nu * nu, 0.0);
            {
              nb::gil_scoped_release release;
              self.compute_jk(dm_ptr, opts.compute_j ? vj.data() : nullptr,
                              opts.compute_k ? vk.data() : nullptr, opts);
            }
            return std::make_pair(to_numpy_2d(std::move(vj), nu, nu),
                                  to_numpy_2d(std::move(vk), nu, nu));
          },
          "dm"_a, "opts"_a = gpu4dft::JKOptions{});

  m.def(
      "jk_nuclear_grad",
      [](const gpu4dft::Basis& basis, ConstMat dm, double k_fraction, gpu4dft::Backend backend,
         double schwarz_cutoff, const std::optional<std::vector<gpu4dft::Atom>>& atoms) {
        check_square(dm, basis.nao(), "dm");
        const double* dm_ptr = dm.data();
        std::vector<double> g;
        {
          nb::gil_scoped_release release;
          if (atoms.has_value()) {
            g = gpu4dft::jk_nuclear_grad(basis, *atoms, dm_ptr, k_fraction, backend,
                                         schwarz_cutoff);
          } else {
            g = gpu4dft::jk_nuclear_grad(basis, dm_ptr, k_fraction, backend, schwarz_cutoff);
          }
        }
        const std::size_t natom = g.size() / 3;
        return to_numpy_2d(std::move(g), natom, 3);
      },
      "basis"_a, "dm"_a, "k_fraction"_a, "backend"_a = gpu4dft::Backend::Cuda,
      "schwarz_cutoff"_a = 1e-13, "atoms"_a = nb::none());

  // -------------------------------------------------------------------------
  // MolecularGrid, XcEngine & SemiNumericalK (`sn-K` / COSX)
  // -------------------------------------------------------------------------
  nb::class_<gpu4dft::GridOptions>(m, "GridOptions")
      .def(nb::init<>())
      .def_rw("level", &gpu4dft::GridOptions::level)
      .def_rw("n_rad", &gpu4dft::GridOptions::n_rad)
      .def_rw("n_ang", &gpu4dft::GridOptions::n_ang)
      .def_rw("prune", &gpu4dft::GridOptions::prune)
      .def_rw("partition", &gpu4dft::GridOptions::partition)
      .def_rw("becke_radii_adjust", &gpu4dft::GridOptions::becke_radii_adjust)
      .def_rw("weight_cutoff", &gpu4dft::GridOptions::weight_cutoff)
      .def_rw("max_batch_points", &gpu4dft::GridOptions::max_batch_points)
      .def_rw("ao_cutoff", &gpu4dft::GridOptions::ao_cutoff)
      .def_rw("backend", &gpu4dft::GridOptions::backend);

  nb::class_<gpu4dft::MolecularGrid>(m, "MolecularGrid")
      .def(nb::init<>())
      .def(
          "__init__",
          [](gpu4dft::MolecularGrid* self, const std::vector<gpu4dft::Atom>& atoms,
             const gpu4dft::Basis& basis, const gpu4dft::GridOptions& opts) {
            const auto xc_at = atoms_to_xc(atoms);
            nb::gil_scoped_release release;
            new (self) gpu4dft::MolecularGrid(xc_at, basis, opts);
          },
          "atoms"_a, "basis"_a, "opts"_a = gpu4dft::GridOptions{})
      .def_static(
          "from_points",
          [](ConstMat coords, ConstVec weights, const gpu4dft::Basis& basis, int max_batch_points,
             double ao_cutoff) {
            if (coords.shape(1) != 3 || coords.shape(0) != weights.shape(0)) {
              throw std::invalid_argument(
                  "MolecularGrid.from_points: coords must have shape (N, 3) and weights shape (N,)");
            }
            const std::size_t npts = weights.shape(0);
            const double* c_ptr = coords.data();
            const double* w_ptr = weights.data();
            nb::gil_scoped_release release;
            return gpu4dft::MolecularGrid::from_points(std::span<const double>(c_ptr, npts * 3),
                                                       std::span<const double>(w_ptr, npts), basis,
                                                       max_batch_points, ao_cutoff);
          },
          "coords"_a, "weights"_a, "basis"_a, "max_batch_points"_a = 2048, "ao_cutoff"_a = 1e-12)
      .def_static(
          "lebedev_grid",
          [](int n_ang) {
            auto [xyz, w] = gpu4dft::MolecularGrid::lebedev_grid(n_ang);
            const std::size_t n = w.size();
            return std::make_pair(to_numpy_2d(std::move(xyz), n, 3), to_numpy_1d(std::move(w), n));
          },
          "n_ang"_a)
      .def_prop_ro("npoints", &gpu4dft::MolecularGrid::npoints)
      .def_prop_ro("nbatches", &gpu4dft::MolecularGrid::nbatches)
      .def_prop_ro("nao", &gpu4dft::MolecularGrid::nao)
      .def_prop_ro(
          "coords",
          [](const gpu4dft::MolecularGrid& self) {
            return to_numpy_2d(self.coords(), self.npoints(), 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "weights",
          [](const gpu4dft::MolecularGrid& self) {
            return to_numpy_1d(self.weights(), self.npoints());
          },
          nb::rv_policy::move);

  m.def("xc_exact_exchange_fraction", &gpu4dft::xc_exact_exchange_fraction, "xc_name"_a);

  m.def(
      "eval_ao_grid",
      [](const gpu4dft::Basis& basis, ConstMat coords, int deriv, gpu4dft::Backend backend) {
        if (coords.shape(1) != 3) {
          throw std::invalid_argument("eval_ao_grid: coords must have shape (N, 3)");
        }
        const std::size_t npts = coords.shape(0);
        const double* c_ptr = coords.data();
        std::vector<double> ao;
        {
          nb::gil_scoped_release release;
          ao = gpu4dft::eval_ao_grid(basis, std::span<const double>(c_ptr, npts * 3), deriv,
                                     backend);
        }
        const auto nao = static_cast<std::size_t>(basis.nao());
        if (deriv == 0) {
          return nb::cast(to_numpy_2d(std::move(ao), npts, nao));
        }
        const std::size_t ncomp = (deriv == 1) ? 4 : 10;
        return nb::cast(to_numpy_3d(std::move(ao), ncomp, npts, nao));
      },
      "basis"_a, "coords"_a, "deriv"_a = 0, "backend"_a = gpu4dft::Backend::Host);

  nb::class_<gpu4dft::XcOptions>(m, "XcOptions")
      .def(nb::init<>())
      .def_rw("backend", &gpu4dft::XcOptions::backend)
      .def_rw("precision", &gpu4dft::XcOptions::precision)
      .def_rw("rho_cutoff", &gpu4dft::XcOptions::rho_cutoff)
      .def_rw("cache_ao_on_device", &gpu4dft::XcOptions::cache_ao_on_device)
      .def_rw("max_ao_cache_mb", &gpu4dft::XcOptions::max_ao_cache_mb);

  nb::class_<gpu4dft::XcResult>(m, "XcResult")
      .def_ro("exc", &gpu4dft::XcResult::exc)
      .def_ro("nelec", &gpu4dft::XcResult::nelec)
      .def_prop_ro(
          "vxc",
          [](const gpu4dft::XcResult& self) {
            const auto n = static_cast<std::size_t>(std::sqrt(static_cast<double>(self.vxc.size())));
            return to_numpy_2d(self.vxc, n, n);
          },
          nb::rv_policy::move);

  nb::class_<gpu4dft::XcEngine>(m, "XcEngine")
      .def(
          "__init__",
          [](gpu4dft::XcEngine* self, const gpu4dft::Basis& basis,
             const gpu4dft::MolecularGrid& grid, const std::string& xc_name,
             const gpu4dft::XcOptions& opts) {
            nb::gil_scoped_release release;
            new (self) gpu4dft::XcEngine(basis, grid, xc_name, opts);
          },
          "basis"_a, "grid"_a, "xc_name"_a, "opts"_a = gpu4dft::XcOptions{})
      .def_prop_ro("nao", &gpu4dft::XcEngine::nao)
      .def_prop_ro("xc_name", &gpu4dft::XcEngine::xc_name)
      .def_prop_ro("exact_exchange_fraction", &gpu4dft::XcEngine::exact_exchange_fraction)
      .def_prop_ro("active_backend", &gpu4dft::XcEngine::active_backend)
      .def(
          "eval_vxc",
          [](const gpu4dft::XcEngine& self, ConstMat dm) {
            const int n = self.nao();
            check_square(dm, n, "dm");
            const double* dm_ptr = dm.data();
            const std::size_t n2 = static_cast<std::size_t>(n) * n;
            nb::gil_scoped_release release;
            return self.eval_vxc(std::span<const double>(dm_ptr, n2));
          },
          "dm"_a)
      .def_prop_ro("has_reference_density", &gpu4dft::XcEngine::has_reference_density)
      .def(
          "set_reference_density",
          [](gpu4dft::XcEngine& self, ConstMat dm) {
            const int n = self.nao();
            check_square(dm, n, "dm");
            const double* dm_ptr = dm.data();
            const std::size_t n2 = static_cast<std::size_t>(n) * n;
            nb::gil_scoped_release release;
            self.set_reference_density(std::span<const double>(dm_ptr, n2));
          },
          "dm"_a)
      .def(
          "eval_fxc",
          [](const gpu4dft::XcEngine& self, ConstMat delta_dm) {
            const int n = self.nao();
            check_square(delta_dm, n, "delta_dm");
            const double* dm_ptr = delta_dm.data();
            const std::size_t n2 = static_cast<std::size_t>(n) * n;
            std::vector<double> fxc;
            {
              nb::gil_scoped_release release;
              fxc = self.eval_fxc(std::span<const double>(dm_ptr, n2));
            }
            const auto nu = static_cast<std::size_t>(n);
            return to_numpy_2d(std::move(fxc), nu, nu);
          },
          "delta_dm"_a)
      .def(
          "nuclear_grad",
          [](const gpu4dft::XcEngine& self, const std::vector<gpu4dft::Atom>& atoms, ConstMat dm) {
            const int n = self.nao();
            check_square(dm, n, "dm");
            const auto xc_at = atoms_to_xc(atoms);
            const double* dm_ptr = dm.data();
            const std::size_t n2 = static_cast<std::size_t>(n) * n;
            std::vector<double> g;
            {
              nb::gil_scoped_release release;
              g = self.nuclear_grad(xc_at, std::span<const double>(dm_ptr, n2));
            }
            return to_numpy_2d(std::move(g), atoms.size(), 3);
          },
          "atoms"_a, "dm"_a);

  nb::class_<gpu4dft::SnkOptions>(m, "SnkOptions")
      .def(nb::init<>())
      .def_rw("backend", &gpu4dft::SnkOptions::backend)
      .def_rw("precision", &gpu4dft::SnkOptions::precision)
      .def_rw("overlap_fitting", &gpu4dft::SnkOptions::overlap_fitting)
      .def_rw("pair_cutoff", &gpu4dft::SnkOptions::pair_cutoff)
      .def_rw("lindep_tol", &gpu4dft::SnkOptions::lindep_tol)
      .def_rw("cache_ao_on_device", &gpu4dft::SnkOptions::cache_ao_on_device)
      .def_rw("max_ao_cache_mb", &gpu4dft::SnkOptions::max_ao_cache_mb);

  nb::class_<gpu4dft::SemiNumericalK>(m, "SemiNumericalK")
      .def(
          "__init__",
          [](gpu4dft::SemiNumericalK* self, const gpu4dft::Basis& basis,
             const gpu4dft::MolecularGrid& grid, const gpu4dft::SnkOptions& opts) {
            nb::gil_scoped_release release;
            new (self) gpu4dft::SemiNumericalK(basis, grid, opts);
          },
          "basis"_a, "grid"_a, "opts"_a = gpu4dft::SnkOptions{})
      .def_prop_ro("nao", &gpu4dft::SemiNumericalK::nao)
      .def_prop_ro("active_backend", &gpu4dft::SemiNumericalK::active_backend)
      .def_prop_ro(
          "overlap_fit_matrix",
          [](const gpu4dft::SemiNumericalK& self) {
            const auto n = static_cast<std::size_t>(self.nao());
            return to_numpy_2d(self.overlap_fit_matrix(), n, n);
          },
          nb::rv_policy::move)
      .def(
          "compute_k",
          [](const gpu4dft::SemiNumericalK& self, ConstMat dm) {
            const int n = self.nao();
            check_square(dm, n, "dm");
            const double* dm_ptr = dm.data();
            const std::size_t n2 = static_cast<std::size_t>(n) * n;
            std::vector<double> vk;
            {
              nb::gil_scoped_release release;
              vk = self.compute_k(std::span<const double>(dm_ptr, n2));
            }
            const auto nu = static_cast<std::size_t>(n);
            return to_numpy_2d(std::move(vk), nu, nu);
          },
          "dm"_a);

  // -------------------------------------------------------------------------
  // Linear Algebra, Ozaki INT8 GEMM Benchmark & SP2 / cuSOLVER Solvers
  // -------------------------------------------------------------------------
  nb::class_<gpu4dft::GemmBenchmarkResult>(m, "GemmBenchmarkResult")
      .def_ro("n", &gpu4dft::GemmBenchmarkResult::n)
      .def_ro("fp64_ms", &gpu4dft::GemmBenchmarkResult::fp64_ms)
      .def_ro("fp64_tflops", &gpu4dft::GemmBenchmarkResult::fp64_tflops)
      .def_ro("ozaki_ms", &gpu4dft::GemmBenchmarkResult::ozaki_ms)
      .def_ro("ozaki_tflops", &gpu4dft::GemmBenchmarkResult::ozaki_tflops)
      .def_ro("fp32_ms", &gpu4dft::GemmBenchmarkResult::fp32_ms)
      .def_ro("fp32_tflops", &gpu4dft::GemmBenchmarkResult::fp32_tflops)
      .def_ro("ozaki_rel_error", &gpu4dft::GemmBenchmarkResult::ozaki_rel_error)
      .def_ro("ozaki_max_abs_error", &gpu4dft::GemmBenchmarkResult::ozaki_max_abs_error)
      .def_ro("fp32_rel_error", &gpu4dft::GemmBenchmarkResult::fp32_rel_error);

  m.def(
      "benchmark_gemm",
      [](int n, int warmup_iters, int bench_iters, gpu4dft::Backend backend, int ozaki_slices) {
        nb::gil_scoped_release release;
        return gpu4dft::benchmark_gemm(n, warmup_iters, bench_iters, backend, ozaki_slices);
      },
      "n"_a, "warmup_iters"_a = 2, "bench_iters"_a = 5, "backend"_a = gpu4dft::Backend::Cuda,
      "ozaki_slices"_a = 8);

  m.def(
      "gemm",
      [](ConstMat A, ConstMat B, bool transA, bool transB, double alpha, double beta,
         gpu4dft::Backend backend, gpu4dft::GemmMode mode, int ozaki_slices) {
        const int m_dim = static_cast<int>(transA ? A.shape(1) : A.shape(0));
        const int k_a = static_cast<int>(transA ? A.shape(0) : A.shape(1));
        const int k_b = static_cast<int>(transB ? B.shape(1) : B.shape(0));
        const int n_dim = static_cast<int>(transB ? B.shape(0) : B.shape(1));
        if (k_a != k_b) {
          throw std::invalid_argument("gemm: inner matrix dimensions do not match");
        }
        const double* a_ptr = A.data();
        const double* b_ptr = B.data();
        std::vector<double> C(static_cast<std::size_t>(m_dim) * n_dim, 0.0);
        {
          nb::gil_scoped_release release;
          gpu4dft::gemm(m_dim, n_dim, k_a, alpha, a_ptr, transA, b_ptr, transB, beta, C.data(),
                        backend, mode, ozaki_slices);
        }
        return to_numpy_2d(std::move(C), static_cast<std::size_t>(m_dim),
                           static_cast<std::size_t>(n_dim));
      },
      "A"_a, "B"_a, "transA"_a = false, "transB"_a = false, "alpha"_a = 1.0, "beta"_a = 0.0,
      "backend"_a = gpu4dft::Backend::Cuda, "mode"_a = gpu4dft::GemmMode::FP64,
      "ozaki_slices"_a = 8);

  nb::class_<gpu4dft::Sp2Options>(m, "Sp2Options")
      .def(nb::init<>())
      .def_rw("gemm_mode", &gpu4dft::Sp2Options::gemm_mode)
      .def_rw("max_iter", &gpu4dft::Sp2Options::max_iter)
      .def_rw("graph_block_steps", &gpu4dft::Sp2Options::graph_block_steps)
      .def_rw("idempotency_tol", &gpu4dft::Sp2Options::idempotency_tol)
      .def_rw("trace_tol", &gpu4dft::Sp2Options::trace_tol)
      .def_rw("use_cuda_graph", &gpu4dft::Sp2Options::use_cuda_graph)
      .def_rw("mcweeny_polish", &gpu4dft::Sp2Options::mcweeny_polish)
      .def_rw("ozaki_slices", &gpu4dft::Sp2Options::ozaki_slices);

  // -------------------------------------------------------------------------
  // C++20 SCF Driver & End-to-End Molecular Workflow
  // -------------------------------------------------------------------------
  nb::class_<gpu4dft::ScfOptions>(m, "ScfOptions")
      .def(nb::init<>())
      .def_rw("backend", &gpu4dft::ScfOptions::backend)
      .def_rw("solver", &gpu4dft::ScfOptions::solver)
      .def_rw("gemm_mode", &gpu4dft::ScfOptions::gemm_mode)
      .def_rw("early_gemm_mode", &gpu4dft::ScfOptions::early_gemm_mode)
      .def_rw("early_integral_precision", &gpu4dft::ScfOptions::early_integral_precision)
      .def_rw("max_iter", &gpu4dft::ScfOptions::max_iter)
      .def_rw("conv_tol", &gpu4dft::ScfOptions::conv_tol)
      .def_rw("diis_tol", &gpu4dft::ScfOptions::diis_tol)
      .def_rw("diis_max_vec", &gpu4dft::ScfOptions::diis_max_vec)
      .def_rw("damping", &gpu4dft::ScfOptions::damping)
      .def_rw("damping_steps", &gpu4dft::ScfOptions::damping_steps)
      .def_rw("level_shift", &gpu4dft::ScfOptions::level_shift)
      .def_rw("lindep_tol", &gpu4dft::ScfOptions::lindep_tol)
      .def_rw("incremental_fock", &gpu4dft::ScfOptions::incremental_fock)
      .def_rw("fock_reset_interval", &gpu4dft::ScfOptions::fock_reset_interval)
      .def_rw("dynamic_precision", &gpu4dft::ScfOptions::dynamic_precision)
      .def_rw("fp64_switch_rms", &gpu4dft::ScfOptions::fp64_switch_rms)
      .def_rw("use_cuda_graphs", &gpu4dft::ScfOptions::use_cuda_graphs)
      .def_rw("exact_exchange_fraction", &gpu4dft::ScfOptions::exact_exchange_fraction)
      .def_rw("ozaki_slices", &gpu4dft::ScfOptions::ozaki_slices)
      .def_rw("sp2_options", &gpu4dft::ScfOptions::sp2_options);

  nb::class_<gpu4dft::ScfIterationLog>(m, "ScfIterationLog")
      .def_ro("iteration", &gpu4dft::ScfIterationLog::iteration)
      .def_ro("total_energy", &gpu4dft::ScfIterationLog::total_energy)
      .def_ro("delta_energy", &gpu4dft::ScfIterationLog::delta_energy)
      .def_ro("diis_rms", &gpu4dft::ScfIterationLog::diis_rms)
      .def_ro("incremental_build", &gpu4dft::ScfIterationLog::incremental_build)
      .def_ro("integral_precision", &gpu4dft::ScfIterationLog::integral_precision)
      .def_ro("gemm_mode", &gpu4dft::ScfIterationLog::gemm_mode);

  nb::class_<gpu4dft::ScfResult>(m, "ScfResult")
      .def_ro("converged", &gpu4dft::ScfResult::converged)
      .def_ro("iterations", &gpu4dft::ScfResult::iterations)
      .def_ro("nao", &gpu4dft::ScfResult::nao)
      .def_ro("nmo", &gpu4dft::ScfResult::nmo)
      .def_ro("total_energy", &gpu4dft::ScfResult::total_energy)
      .def_ro("electronic_energy", &gpu4dft::ScfResult::electronic_energy)
      .def_ro("one_electron_energy", &gpu4dft::ScfResult::one_electron_energy)
      .def_ro("two_electron_energy", &gpu4dft::ScfResult::two_electron_energy)
      .def_ro("xc_energy", &gpu4dft::ScfResult::xc_energy)
      .def_ro("nuclear_repulsion", &gpu4dft::ScfResult::nuclear_repulsion)
      .def_ro("final_diis_rms", &gpu4dft::ScfResult::final_diis_rms)
      .def_ro("cuda_graph_captures", &gpu4dft::ScfResult::cuda_graph_captures)
      .def_ro("cuda_graph_replays", &gpu4dft::ScfResult::cuda_graph_replays)
      .def_ro("pool_cache_hits", &gpu4dft::ScfResult::pool_cache_hits)
      .def_ro("steady_state_allocations", &gpu4dft::ScfResult::steady_state_allocations)
      .def_ro("history", &gpu4dft::ScfResult::history)
      .def_prop_ro(
          "density_matrix",
          [](const gpu4dft::ScfResult& self) {
            const auto n = static_cast<std::size_t>(self.nao);
            return to_numpy_2d(self.density_matrix, n, n);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "energy_weighted_dm",
          [](const gpu4dft::ScfResult& self) {
            const auto n = static_cast<std::size_t>(self.nao);
            return to_numpy_2d(self.energy_weighted_dm, n, n);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "fock_matrix",
          [](const gpu4dft::ScfResult& self) {
            const auto n = static_cast<std::size_t>(self.nao);
            return to_numpy_2d(self.fock_matrix, n, n);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "mo_energies",
          [](const gpu4dft::ScfResult& self) {
            return to_numpy_1d(self.mo_energies, self.mo_energies.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "mo_coefficients",
          [](const gpu4dft::ScfResult& self) {
            if (self.mo_coefficients.empty()) {
              return to_numpy_2d({}, 0, 0);
            }
            return to_numpy_2d(self.mo_coefficients, static_cast<std::size_t>(self.nao),
                               static_cast<std::size_t>(self.nmo));
          },
          nb::rv_policy::move);

  nb::class_<gpu4dft::WorkflowOptions>(m, "WorkflowOptions")
      .def(nb::init<>())
      .def_rw("xc", &gpu4dft::WorkflowOptions::xc)
      .def_rw("jk_mode", &gpu4dft::WorkflowOptions::jk_mode)
      .def_rw("scf", &gpu4dft::WorkflowOptions::scf)
      .def_rw("schwarz_cutoff", &gpu4dft::WorkflowOptions::schwarz_cutoff)
      .def_rw("fp32_threshold", &gpu4dft::WorkflowOptions::fp32_threshold)
      .def_rw("prim_cutoff", &gpu4dft::WorkflowOptions::prim_cutoff)
      .def_rw("grid", &gpu4dft::WorkflowOptions::grid)
      .def_rw("xc_options", &gpu4dft::WorkflowOptions::xc_options)
      .def_rw("snk_options", &gpu4dft::WorkflowOptions::snk_options);

  nb::class_<gpu4dft::WorkflowResult>(m, "WorkflowResult")
      .def_ro("scf", &gpu4dft::WorkflowResult::scf)
      .def_ro("setup_time_s", &gpu4dft::WorkflowResult::setup_time_s)
      .def_ro("one_electron_time_s", &gpu4dft::WorkflowResult::one_electron_time_s)
      .def_ro("jk_setup_time_s", &gpu4dft::WorkflowResult::jk_setup_time_s)
      .def_ro("grid_setup_time_s", &gpu4dft::WorkflowResult::grid_setup_time_s)
      .def_ro("scf_time_s", &gpu4dft::WorkflowResult::scf_time_s)
      .def_ro("grid_npoints", &gpu4dft::WorkflowResult::grid_npoints)
      .def_ro("grid_nbatches", &gpu4dft::WorkflowResult::grid_nbatches)
      .def_ro("active_ao_ratio", &gpu4dft::WorkflowResult::active_ao_ratio)
      .def_prop_ro(
          "overlap_matrix",
          [](const gpu4dft::WorkflowResult& self) {
            const auto n = static_cast<std::size_t>(self.scf.nao);
            return to_numpy_2d(self.overlap_matrix, n, n);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "hcore_matrix",
          [](const gpu4dft::WorkflowResult& self) {
            const auto n = static_cast<std::size_t>(self.scf.nao);
            return to_numpy_2d(self.hcore_matrix, n, n);
          },
          nb::rv_policy::move);

  nb::class_<gpu4dft::GradientResult>(m, "GradientResult")
      .def_ro("natom", &gpu4dft::GradientResult::natom)
      .def_ro("total_time_s", &gpu4dft::GradientResult::total_time_s)
      .def_ro("one_electron_time_s", &gpu4dft::GradientResult::one_electron_time_s)
      .def_ro("two_electron_time_s", &gpu4dft::GradientResult::two_electron_time_s)
      .def_ro("xc_time_s", &gpu4dft::GradientResult::xc_time_s)
      .def_prop_ro(
          "total_grad",
          [](const gpu4dft::GradientResult& self) {
            return to_numpy_2d(self.total_grad, static_cast<std::size_t>(self.natom), 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "nuclear_repulsion_grad",
          [](const gpu4dft::GradientResult& self) {
            return to_numpy_2d(self.nuclear_repulsion_grad, static_cast<std::size_t>(self.natom),
                               3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "one_electron_grad",
          [](const gpu4dft::GradientResult& self) {
            return to_numpy_2d(self.one_electron_grad, static_cast<std::size_t>(self.natom), 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "two_electron_grad",
          [](const gpu4dft::GradientResult& self) {
            return to_numpy_2d(self.two_electron_grad, static_cast<std::size_t>(self.natom), 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "xc_grad",
          [](const gpu4dft::GradientResult& self) {
            return to_numpy_2d(self.xc_grad, static_cast<std::size_t>(self.natom), 3);
          },
          nb::rv_policy::move);

  // -------------------------------------------------------------------------
  // Electric Dipoles, Excited-State Linear Response (CIS/TDA/TDHF/TDDFT) & Hessian
  // -------------------------------------------------------------------------
  m.def(
      "dipole_matrices",
      [](const gpu4dft::Basis& basis, std::array<double, 3> origin, gpu4dft::Backend backend) {
        std::array<std::vector<double>, 3> dip;
        {
          nb::gil_scoped_release release;
          dip = gpu4dft::dipole_matrices(basis, origin, backend);
        }
        const auto n = static_cast<std::size_t>(basis.nao());
        const std::size_t n2 = n * n;
        std::vector<double> flat(3 * n2, 0.0);
        for (std::size_t c = 0; c < 3; ++c) {
          std::copy(dip[c].begin(), dip[c].end(), flat.begin() + static_cast<std::ptrdiff_t>(c * n2));
        }
        return to_numpy_3d(std::move(flat), 3, n, n);
      },
      "basis"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0},
      "backend"_a = gpu4dft::Backend::Host);

  m.def(
      "molecular_dipole_moment",
      [](const gpu4dft::Basis& basis, const std::vector<gpu4dft::Atom>& atoms, ConstMat dm,
         std::array<double, 3> origin, gpu4dft::Backend backend) {
        check_square(dm, basis.nao(), "dm");
        const double* dm_ptr = dm.data();
        std::array<double, 3> mu{};
        {
          nb::gil_scoped_release release;
          mu = gpu4dft::molecular_dipole_moment(basis, atoms, dm_ptr, origin, backend);
        }
        return to_numpy_1d({mu[0], mu[1], mu[2]}, 3);
      },
      "basis"_a, "atoms"_a, "dm"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0},
      "backend"_a = gpu4dft::Backend::Host);

  nb::class_<gpu4dft::ResponseOptions>(m, "ResponseOptions")
      .def(nb::init<>())
      .def_rw("backend", &gpu4dft::ResponseOptions::backend)
      .def_rw("method", &gpu4dft::ResponseOptions::method)
      .def_rw("gemm_mode", &gpu4dft::ResponseOptions::gemm_mode)
      .def_rw("dynamic_precision", &gpu4dft::ResponseOptions::dynamic_precision)
      .def_rw("early_gemm_mode", &gpu4dft::ResponseOptions::early_gemm_mode)
      .def_rw("fp64_switch_residual", &gpu4dft::ResponseOptions::fp64_switch_residual)
      .def_rw("ozaki_slices", &gpu4dft::ResponseOptions::ozaki_slices)
      .def_rw("n_states", &gpu4dft::ResponseOptions::n_states)
      .def_rw("max_iter", &gpu4dft::ResponseOptions::max_iter)
      .def_rw("max_subspace", &gpu4dft::ResponseOptions::max_subspace)
      .def_rw("conv_tol", &gpu4dft::ResponseOptions::conv_tol)
      .def_rw("exact_exchange_fraction", &gpu4dft::ResponseOptions::exact_exchange_fraction)
      .def_rw("denom_floor", &gpu4dft::ResponseOptions::denom_floor);

  nb::class_<gpu4dft::DavidsonIterationLog>(m, "DavidsonIterationLog")
      .def_ro("iteration", &gpu4dft::DavidsonIterationLog::iteration)
      .def_ro("subspace_dim", &gpu4dft::DavidsonIterationLog::subspace_dim)
      .def_ro("max_residual", &gpu4dft::DavidsonIterationLog::max_residual)
      .def_ro("gemm_mode", &gpu4dft::DavidsonIterationLog::gemm_mode);

  nb::class_<gpu4dft::ExcitedState>(m, "ExcitedState")
      .def_ro("state_index", &gpu4dft::ExcitedState::state_index)
      .def_ro("energy_hartree", &gpu4dft::ExcitedState::energy_hartree)
      .def_ro("energy_ev", &gpu4dft::ExcitedState::energy_ev)
      .def_ro("transition_dipole", &gpu4dft::ExcitedState::transition_dipole)
      .def_ro("oscillator_strength", &gpu4dft::ExcitedState::oscillator_strength)
      .def_ro("residual_norm", &gpu4dft::ExcitedState::residual_norm)
      .def_ro("dominant_occ", &gpu4dft::ExcitedState::dominant_occ)
      .def_ro("dominant_vir", &gpu4dft::ExcitedState::dominant_vir)
      .def_ro("dominant_weight", &gpu4dft::ExcitedState::dominant_weight);

  nb::class_<gpu4dft::ResponseResult>(m, "ResponseResult")
      .def_ro("converged", &gpu4dft::ResponseResult::converged)
      .def_ro("iterations", &gpu4dft::ResponseResult::iterations)
      .def_ro("n_states", &gpu4dft::ResponseResult::n_states)
      .def_ro("nao", &gpu4dft::ResponseResult::nao)
      .def_ro("nmo", &gpu4dft::ResponseResult::nmo)
      .def_ro("n_occ", &gpu4dft::ResponseResult::n_occ)
      .def_ro("n_vir", &gpu4dft::ResponseResult::n_vir)
      .def_ro("matvec_count", &gpu4dft::ResponseResult::matvec_count)
      .def_ro("wall_time_s", &gpu4dft::ResponseResult::wall_time_s)
      .def_ro("states", &gpu4dft::ResponseResult::states)
      .def_ro("history", &gpu4dft::ResponseResult::history)
      .def_prop_ro(
          "excitation_energies",
          [](const gpu4dft::ResponseResult& self) {
            return to_numpy_1d(self.excitation_energies, self.excitation_energies.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "excitation_energies_ev",
          [](const gpu4dft::ResponseResult& self) {
            return to_numpy_1d(self.excitation_energies_ev, self.excitation_energies_ev.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "oscillator_strengths",
          [](const gpu4dft::ResponseResult& self) {
            return to_numpy_1d(self.oscillator_strengths, self.oscillator_strengths.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "transition_dipoles",
          [](const gpu4dft::ResponseResult& self) {
            const std::size_t ns = self.transition_dipoles.size();
            std::vector<double> flat(ns * 3, 0.0);
            for (std::size_t k = 0; k < ns; ++k) {
              flat[3 * k + 0] = self.transition_dipoles[k][0];
              flat[3 * k + 1] = self.transition_dipoles[k][1];
              flat[3 * k + 2] = self.transition_dipoles[k][2];
            }
            return to_numpy_2d(std::move(flat), ns, 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "x_vectors",
          [](const gpu4dft::ResponseResult& self) {
            return to_numpy_3d(self.x_vectors, static_cast<std::size_t>(self.n_states),
                               static_cast<std::size_t>(self.n_occ),
                               static_cast<std::size_t>(self.n_vir));
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "y_vectors",
          [](const gpu4dft::ResponseResult& self) {
            return to_numpy_3d(self.y_vectors, static_cast<std::size_t>(self.n_states),
                               static_cast<std::size_t>(self.n_occ),
                               static_cast<std::size_t>(self.n_vir));
          },
          nb::rv_policy::move);

  nb::class_<gpu4dft::HessianOptions>(m, "HessianOptions")
      .def(nb::init<>())
      .def_rw("backend", &gpu4dft::HessianOptions::backend)
      .def_rw("step_bohr", &gpu4dft::HessianOptions::step_bohr)
      .def_rw("project_trans_rot", &gpu4dft::HessianOptions::project_trans_rot)
      .def_rw("tr_svd_tol", &gpu4dft::HessianOptions::tr_svd_tol)
      .def_rw("fixed_xc_grid", &gpu4dft::HessianOptions::fixed_xc_grid);

  nb::class_<gpu4dft::VibrationalResult>(m, "VibrationalResult")
      .def_ro("natom", &gpu4dft::VibrationalResult::natom)
      .def_ro("n_vib", &gpu4dft::VibrationalResult::n_vib)
      .def_ro("n_trans_rot", &gpu4dft::VibrationalResult::n_trans_rot)
      .def_ro("imaginary_count", &gpu4dft::VibrationalResult::imaginary_count)
      .def_ro("zero_point_energy_hartree", &gpu4dft::VibrationalResult::zero_point_energy_hartree)
      .def_ro("wall_time_s", &gpu4dft::VibrationalResult::wall_time_s)
      .def_prop_ro(
          "hessian",
          [](const gpu4dft::VibrationalResult& self) {
            const auto n3 = static_cast<std::size_t>(3 * self.natom);
            return to_numpy_2d(self.hessian, n3, n3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "frequencies_cm1",
          [](const gpu4dft::VibrationalResult& self) {
            return to_numpy_1d(self.frequencies_cm1, self.frequencies_cm1.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "force_constants_au",
          [](const gpu4dft::VibrationalResult& self) {
            return to_numpy_1d(self.force_constants_au, self.force_constants_au.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "reduced_masses_amu",
          [](const gpu4dft::VibrationalResult& self) {
            return to_numpy_1d(self.reduced_masses_amu, self.reduced_masses_amu.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "normal_modes",
          [](const gpu4dft::VibrationalResult& self) {
            return to_numpy_3d(self.normal_modes, static_cast<std::size_t>(self.n_vib),
                               static_cast<std::size_t>(self.natom), 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "mass_weighted_modes",
          [](const gpu4dft::VibrationalResult& self) {
            return to_numpy_2d(self.mass_weighted_modes, static_cast<std::size_t>(self.n_vib),
                               static_cast<std::size_t>(3 * self.natom));
          },
          nb::rv_policy::move);

  m.def("standard_atomic_mass_amu", &gpu4dft::standard_atomic_mass_amu, "atomic_number"_a);

  m.def(
      "analyze_hessian",
      [](const std::vector<gpu4dft::Atom>& atoms, ConstMat hessian_2d,
         std::optional<ConstVec> masses, const gpu4dft::HessianOptions& opts) {
        const int n3 = static_cast<int>(3 * atoms.size());
        check_square(hessian_2d, n3, "hessian_2d");
        const double* h_ptr = hessian_2d.data();
        std::span<const double> m_span;
        if (masses.has_value()) {
          if (masses->shape(0) != atoms.size()) {
            throw std::invalid_argument("analyze_hessian: masses length must equal len(atoms)");
          }
          m_span = std::span<const double>(masses->data(), atoms.size());
        }
        nb::gil_scoped_release release;
        return gpu4dft::analyze_hessian(atoms, h_ptr, m_span, opts);
      },
      "atoms"_a, "hessian_2d"_a, "masses"_a = nb::none(),
      "opts"_a = gpu4dft::HessianOptions{});

  m.def(
      "run_linear_response",
      [](const gpu4dft::Basis& basis, const std::vector<gpu4dft::Atom>& atoms, int n_occ,
         const gpu4dft::ScfResult& scf_res, const std::string& xc_name,
         const gpu4dft::ResponseOptions& opts, std::optional<ConstMat> grid_coords,
         std::optional<ConstVec> grid_weights, const gpu4dft::GridOptions& grid_opts) {
        std::span<const double> c_span;
        std::span<const double> w_span;
        if (grid_coords.has_value() || grid_weights.has_value()) {
          if (!grid_coords.has_value() || !grid_weights.has_value()) {
            throw std::invalid_argument(
                "run_linear_response: grid_coords and grid_weights must both be provided");
          }
          if (grid_coords->shape(1) != 3 || grid_coords->shape(0) != grid_weights->shape(0)) {
            throw std::invalid_argument(
                "run_linear_response: grid_coords must be (N, 3) and grid_weights must be (N,)");
          }
          const std::size_t npts = grid_weights->shape(0);
          c_span = std::span<const double>(grid_coords->data(), npts * 3);
          w_span = std::span<const double>(grid_weights->data(), npts);
        }
        nb::gil_scoped_release release;
        return gpu4dft::run_linear_response(basis, atoms, n_occ, scf_res, xc_name, opts, c_span,
                                            w_span, grid_opts);
      },
      "basis"_a, "atoms"_a, "n_occ"_a, "scf_res"_a, "xc_name"_a = "hf",
      "opts"_a = gpu4dft::ResponseOptions{}, "grid_coords"_a = nb::none(),
      "grid_weights"_a = nb::none(), "grid_opts"_a = gpu4dft::GridOptions{});

  m.def(
      "compute_molecular_hessian",
      [](const gpu4dft::Basis& basis, const std::vector<gpu4dft::Atom>& atoms, int n_elec,
         const gpu4dft::WorkflowOptions& wf_opts, std::optional<ConstVec> masses,
         const gpu4dft::HessianOptions& hess_opts, std::optional<ConstMat> grid_coords,
         std::optional<ConstVec> grid_weights) {
        std::span<const double> m_span;
        if (masses.has_value()) {
          if (masses->shape(0) != atoms.size()) {
            throw std::invalid_argument(
                "compute_molecular_hessian: masses length must equal len(atoms)");
          }
          m_span = std::span<const double>(masses->data(), atoms.size());
        }
        std::span<const double> c_span;
        std::span<const double> w_span;
        if (grid_coords.has_value() || grid_weights.has_value()) {
          if (!grid_coords.has_value() || !grid_weights.has_value()) {
            throw std::invalid_argument(
                "compute_molecular_hessian: grid_coords and grid_weights must both be provided");
          }
          if (grid_coords->shape(1) != 3 || grid_coords->shape(0) != grid_weights->shape(0)) {
            throw std::invalid_argument(
                "compute_molecular_hessian: grid_coords must be (N, 3) and grid_weights must be (N,)");
          }
          const std::size_t npts = grid_weights->shape(0);
          c_span = std::span<const double>(grid_coords->data(), npts * 3);
          w_span = std::span<const double>(grid_weights->data(), npts);
        }
        nb::gil_scoped_release release;
        return gpu4dft::compute_molecular_hessian(basis, atoms, n_elec, wf_opts, m_span, hess_opts,
                                                  c_span, w_span);
      },
      "basis"_a, "atoms"_a, "n_elec"_a, "wf_opts"_a = gpu4dft::WorkflowOptions{},
      "masses"_a = nb::none(), "hess_opts"_a = gpu4dft::HessianOptions{},
      "grid_coords"_a = nb::none(), "grid_weights"_a = nb::none());

  nb::class_<gpu4dft::MolecularScf>(m, "MolecularScf")
      .def(
          "__init__",
          [](gpu4dft::MolecularScf* self, const gpu4dft::Basis& basis,
             std::vector<gpu4dft::Atom> atoms, int n_elec, const gpu4dft::WorkflowOptions& opts,
             std::optional<gpu4dft::Basis> aux_basis, std::optional<ConstMat> grid_coords,
             std::optional<ConstVec> grid_weights) {
            std::span<const double> c_span;
            std::span<const double> w_span;
            if (grid_coords.has_value() || grid_weights.has_value()) {
              if (!grid_coords.has_value() || !grid_weights.has_value()) {
                throw std::invalid_argument(
                    "MolecularScf: grid_coords and grid_weights must both be provided");
              }
              if (grid_coords->shape(1) != 3 ||
                  grid_coords->shape(0) != grid_weights->shape(0)) {
                throw std::invalid_argument(
                    "MolecularScf: grid_coords must be (N, 3) and grid_weights must be (N,)");
              }
              const std::size_t npts = grid_weights->shape(0);
              c_span = std::span<const double>(grid_coords->data(), npts * 3);
              w_span = std::span<const double>(grid_weights->data(), npts);
            }
            new (self) gpu4dft::MolecularScf(basis, std::move(atoms), n_elec, opts,
                                             std::move(aux_basis), c_span, w_span);
          },
          "basis"_a, "atoms"_a, "n_elec"_a, "opts"_a = gpu4dft::WorkflowOptions{},
          "aux_basis"_a = nb::none(), "grid_coords"_a = nb::none(), "grid_weights"_a = nb::none())
      .def_prop_ro("nao", &gpu4dft::MolecularScf::nao)
      .def_prop_ro("natom", &gpu4dft::MolecularScf::natom)
      .def_prop_ro("n_elec", &gpu4dft::MolecularScf::n_elec)
      .def_prop_ro("is_dft", &gpu4dft::MolecularScf::is_dft)
      .def_prop_ro("exact_exchange_fraction", &gpu4dft::MolecularScf::exact_exchange_fraction)
      .def(
          "setup",
          [](gpu4dft::MolecularScf& self) {
            nb::gil_scoped_release release;
            self.setup();
          })
      .def(
          "run",
          [](gpu4dft::MolecularScf& self, std::optional<ConstMat> dm0) {
            std::span<const double> dm_span;
            if (dm0.has_value()) {
              check_square(*dm0, self.nao(), "dm0");
              dm_span = std::span<const double>(
                  dm0->data(), static_cast<std::size_t>(self.nao()) * self.nao());
            }
            nb::gil_scoped_release release;
            return self.run(dm_span);
          },
          "dm0"_a = nb::none())
      .def(
          "compute_gradient",
          [](gpu4dft::MolecularScf& self) {
            nb::gil_scoped_release release;
            return self.compute_gradient();
          })
      .def(
          "compute_gradient_from",
          [](gpu4dft::MolecularScf& self, const gpu4dft::ScfResult& scf_res) {
            nb::gil_scoped_release release;
            return self.compute_gradient(scf_res);
          },
          "scf_res"_a)
      .def(
          "compute_dipole",
          [](gpu4dft::MolecularScf& self, std::array<double, 3> origin) {
            std::array<double, 3> mu{};
            {
              nb::gil_scoped_release release;
              mu = self.compute_dipole(origin);
            }
            return to_numpy_1d({mu[0], mu[1], mu[2]}, 3);
          },
          "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0})
      .def(
          "compute_dipole_from",
          [](gpu4dft::MolecularScf& self, const gpu4dft::ScfResult& scf_res,
             std::array<double, 3> origin) {
            std::array<double, 3> mu{};
            {
              nb::gil_scoped_release release;
              mu = self.compute_dipole(scf_res, origin);
            }
            return to_numpy_1d({mu[0], mu[1], mu[2]}, 3);
          },
          "scf_res"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0})
      .def(
          "run_response",
          [](gpu4dft::MolecularScf& self, std::optional<gpu4dft::ResponseOptions> opts) {
            nb::gil_scoped_release release;
            if (opts.has_value()) {
              return self.run_response(*opts);
            }
            return self.run_response();
          },
          "opts"_a = nb::none())
      .def(
          "run_hessian",
          [](gpu4dft::MolecularScf& self, std::optional<gpu4dft::HessianOptions> opts,
             std::optional<ConstVec> masses) {
            std::span<const double> m_span;
            if (masses.has_value()) {
              if (static_cast<int>(masses->shape(0)) != self.natom()) {
                throw std::invalid_argument(
                    "MolecularScf.run_hessian: masses length must equal natom");
              }
              m_span = std::span<const double>(masses->data(), static_cast<std::size_t>(self.natom()));
            }
            nb::gil_scoped_release release;
            if (opts.has_value()) {
              return self.run_hessian(*opts, m_span);
            }
            if (masses.has_value()) {
              gpu4dft::HessianOptions hopts;
              hopts.backend = self.options().scf.backend;
              return self.run_hessian(hopts, m_span);
            }
            return self.run_hessian();
          },
          "opts"_a = nb::none(), "masses"_a = nb::none());

  m.def(
      "run_scf",
      [](const gpu4dft::Basis& basis, const std::vector<gpu4dft::Atom>& atoms, int n_elec,
         const gpu4dft::WorkflowOptions& opts, std::optional<ConstMat> dm0,
         std::optional<gpu4dft::Basis> aux_basis, std::optional<ConstMat> grid_coords,
         std::optional<ConstVec> grid_weights) {
        std::span<const double> dm_span;
        if (dm0.has_value()) {
          check_square(*dm0, basis.nao(), "dm0");
          dm_span =
              std::span<const double>(dm0->data(), static_cast<std::size_t>(basis.nao()) * basis.nao());
        }
        std::span<const double> c_span;
        std::span<const double> w_span;
        if (grid_coords.has_value() || grid_weights.has_value()) {
          if (!grid_coords.has_value() || !grid_weights.has_value()) {
            throw std::invalid_argument(
                "run_scf: grid_coords and grid_weights must both be provided");
          }
          if (grid_coords->shape(1) != 3 || grid_coords->shape(0) != grid_weights->shape(0)) {
            throw std::invalid_argument(
                "run_scf: grid_coords must be (N, 3) and grid_weights must be (N,)");
          }
          const std::size_t npts = grid_weights->shape(0);
          c_span = std::span<const double>(grid_coords->data(), npts * 3);
          w_span = std::span<const double>(grid_weights->data(), npts);
        }
        const gpu4dft::Basis* aux_ptr = aux_basis.has_value() ? &(*aux_basis) : nullptr;
        nb::gil_scoped_release release;
        return gpu4dft::run_scf(basis, atoms, n_elec, opts, dm_span, aux_ptr, c_span, w_span);
      },
      "basis"_a, "atoms"_a, "n_elec"_a, "opts"_a = gpu4dft::WorkflowOptions{},
      "dm0"_a = nb::none(), "aux_basis"_a = nb::none(), "grid_coords"_a = nb::none(),
      "grid_weights"_a = nb::none());

  m.def(
      "run_grad",
      [](const gpu4dft::Basis& basis, const std::vector<gpu4dft::Atom>& atoms,
         const gpu4dft::ScfResult& scf_res, const gpu4dft::WorkflowOptions& opts,
         std::optional<ConstMat> grid_coords, std::optional<ConstVec> grid_weights) {
        std::span<const double> c_span;
        std::span<const double> w_span;
        if (grid_coords.has_value() || grid_weights.has_value()) {
          if (!grid_coords.has_value() || !grid_weights.has_value()) {
            throw std::invalid_argument(
                "run_grad: grid_coords and grid_weights must both be provided");
          }
          if (grid_coords->shape(1) != 3 || grid_coords->shape(0) != grid_weights->shape(0)) {
            throw std::invalid_argument(
                "run_grad: grid_coords must be (N, 3) and grid_weights must be (N,)");
          }
          const std::size_t npts = grid_weights->shape(0);
          c_span = std::span<const double>(grid_coords->data(), npts * 3);
          w_span = std::span<const double>(grid_weights->data(), npts);
        }
        nb::gil_scoped_release release;
        return gpu4dft::run_grad(basis, atoms, scf_res, opts, c_span, w_span);
      },
      "basis"_a, "atoms"_a, "scf_res"_a, "opts"_a = gpu4dft::WorkflowOptions{},
      "grid_coords"_a = nb::none(), "grid_weights"_a = nb::none());

  // -------------------------------------------------------------------------
  // Plane-Wave Kohn-Sham & Hybrid ACE DFT Submodule (`gpu4dft._core.pw`)
  // -------------------------------------------------------------------------
  auto pw_m = m.def_submodule("pw", "Plane-Wave Kohn-Sham & Hybrid ACE DFT Engine");

  nb::enum_<gpu4dft::pw::PwSolverType>(pw_m, "PwSolverType")
      .value("davidson", gpu4dft::pw::PwSolverType::Davidson)
      .value("lobpcg", gpu4dft::pw::PwSolverType::LOBPCG);

  nb::enum_<gpu4dft::pw::ExxDiv>(pw_m, "ExxDiv")
      .value("none", gpu4dft::pw::ExxDiv::None)
      .value("ewald", gpu4dft::pw::ExxDiv::Ewald);

  pw_m.def("cuda_pw_available", &gpu4dft::pw::cuda_pw_available);
  pw_m.def("next_fft_smooth_size", &gpu4dft::pw::next_fft_smooth_size, "n"_a);
  pw_m.def(
      "element_atomic_number",
      [](const std::string& symbol) { return gpu4dft::pw::element_atomic_number(symbol); },
      "symbol"_a);
  pw_m.def("element_symbol", &gpu4dft::pw::element_symbol, "atomic_number"_a);

  nb::class_<gpu4dft::pw::GthNonLocalChannel>(pw_m, "GthNonLocalChannel")
      .def(nb::init<>())
      .def_rw("l", &gpu4dft::pw::GthNonLocalChannel::l)
      .def_rw("r_l", &gpu4dft::pw::GthNonLocalChannel::r_l)
      .def_rw("n_proj", &gpu4dft::pw::GthNonLocalChannel::n_proj)
      .def_rw("h", &gpu4dft::pw::GthNonLocalChannel::h);

  nb::class_<gpu4dft::pw::GthPseudopotential>(pw_m, "GthPseudopotential")
      .def(nb::init<>())
      .def_rw("symbol", &gpu4dft::pw::GthPseudopotential::symbol)
      .def_rw("atomic_number", &gpu4dft::pw::GthPseudopotential::atomic_number)
      .def_rw("z_ion", &gpu4dft::pw::GthPseudopotential::z_ion)
      .def_rw("r_loc", &gpu4dft::pw::GthPseudopotential::r_loc)
      .def_rw("c_loc", &gpu4dft::pw::GthPseudopotential::c_loc)
      .def_rw("nl_channels", &gpu4dft::pw::GthPseudopotential::nl_channels)
      .def("core_energy_constant", &gpu4dft::pw::GthPseudopotential::core_energy_constant);

  pw_m.def(
      "lookup_gth_pseudopotential",
      [](int atomic_number, const std::string& family) {
        return gpu4dft::pw::lookup_gth_pseudopotential(atomic_number, family);
      },
      "atomic_number"_a, "family"_a = "gth-pade");

  pw_m.def(
      "lookup_gth_pseudopotential",
      [](const std::string& symbol, const std::string& family) {
        return gpu4dft::pw::lookup_gth_pseudopotential(symbol, family);
      },
      "symbol"_a, "family"_a = "gth-pade");

  nb::class_<gpu4dft::pw::PwAtom>(pw_m, "PwAtom")
      .def(nb::init<>())
      .def(
          "__init__",
          [](gpu4dft::pw::PwAtom* self, int atomic_number, std::array<double, 3> pos) {
            new (self) gpu4dft::pw::PwAtom{atomic_number, {}, pos, {}};
          },
          "atomic_number"_a, "pos"_a)
      .def(
          "__init__",
          [](gpu4dft::pw::PwAtom* self, const std::string& symbol, std::array<double, 3> pos) {
            new (self)
                gpu4dft::pw::PwAtom{gpu4dft::pw::element_atomic_number(symbol), symbol, pos, {}};
          },
          "symbol"_a, "pos"_a)
      .def_rw("atomic_number", &gpu4dft::pw::PwAtom::atomic_number)
      .def_rw("symbol", &gpu4dft::pw::PwAtom::symbol)
      .def_rw("pos", &gpu4dft::pw::PwAtom::pos)
      .def_rw("pseudo", &gpu4dft::pw::PwAtom::pseudo);

  nb::class_<gpu4dft::pw::PwCell>(pw_m, "PwCell")
      .def(nb::init<>())
      .def(
          "__init__",
          [](gpu4dft::pw::PwCell* self, ConstMat a_bohr,
             std::vector<gpu4dft::pw::PwAtom> atoms, const std::string& pseudo_family) {
            new (self) gpu4dft::pw::PwCell(mat_to_lattice(a_bohr), std::move(atoms), pseudo_family);
          },
          "a_bohr"_a, "atoms"_a, "pseudo_family"_a = "gth-pade")
      .def(
          "__init__",
          [](gpu4dft::pw::PwCell* self, ConstMat a_bohr, const std::vector<int>& atomic_numbers,
             ConstMat coords_bohr, const std::string& pseudo_family) {
            new (self) gpu4dft::pw::PwCell(
                mat_to_lattice(a_bohr), make_pw_atoms(atomic_numbers, coords_bohr), pseudo_family);
          },
          "a_bohr"_a, "atomic_numbers"_a, "coords_bohr"_a, "pseudo_family"_a = "gth-pade")
      .def_prop_ro(
          "lattice_vectors",
          [](const gpu4dft::pw::PwCell& self) {
            const auto& a = self.lattice_vectors();
            std::vector<double> flat(9);
            for (std::size_t i = 0; i < 3; ++i)
              for (std::size_t j = 0; j < 3; ++j) flat[3 * i + j] = a[i][j];
            return to_numpy_2d(std::move(flat), 3, 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "reciprocal_vectors",
          [](const gpu4dft::pw::PwCell& self) {
            const auto& b = self.reciprocal_vectors();
            std::vector<double> flat(9);
            for (std::size_t i = 0; i < 3; ++i)
              for (std::size_t j = 0; j < 3; ++j) flat[3 * i + j] = b[i][j];
            return to_numpy_2d(std::move(flat), 3, 3);
          },
          nb::rv_policy::move)
      .def_prop_ro("volume", &gpu4dft::pw::PwCell::volume)
      .def_prop_ro("atoms", &gpu4dft::pw::PwCell::atoms)
      .def_prop_ro("natom", &gpu4dft::pw::PwCell::natom)
      .def_prop_ro("pseudo_family", &gpu4dft::pw::PwCell::pseudo_family)
      .def("total_valence_charge", &gpu4dft::pw::PwCell::total_valence_charge)
      .def("num_electrons", &gpu4dft::pw::PwCell::num_electrons, "charge"_a = 0)
      .def(
          "ewald_energy",
          [](const gpu4dft::pw::PwCell& self, double precision) {
            nb::gil_scoped_release release;
            return self.ewald_energy(precision);
          },
          "precision"_a = 1e-13)
      .def(
          "madelung_constant",
          [](const gpu4dft::pw::PwCell& self, std::array<int, 3> kmesh, double precision) {
            nb::gil_scoped_release release;
            return self.madelung_constant(kmesh, precision);
          },
          "kmesh"_a = std::array<int, 3>{1, 1, 1}, "precision"_a = 1e-13)
      .def(
          "uniform_grid_coords",
          [](const gpu4dft::pw::PwCell& self, std::array<int, 3> mesh) {
            std::vector<double> coords = self.uniform_grid_coords(mesh);
            const std::size_t n = coords.size() / 3;
            return to_numpy_2d(std::move(coords), n, 3);
          },
          "mesh"_a)
      .def(
          "monkhorst_pack_kpoints",
          [](const gpu4dft::pw::PwCell& self, std::array<int, 3> kmesh, bool gamma_centered,
             bool wrap_around) {
            auto [kpts, kw] = self.monkhorst_pack_kpoints(kmesh, gamma_centered, wrap_around);
            const std::size_t nk = kpts.size();
            std::vector<double> kflat(nk * 3, 0.0);
            for (std::size_t i = 0; i < nk; ++i) {
              kflat[3 * i + 0] = kpts[i][0];
              kflat[3 * i + 1] = kpts[i][1];
              kflat[3 * i + 2] = kpts[i][2];
            }
            return std::make_pair(to_numpy_2d(std::move(kflat), nk, 3),
                                  to_numpy_1d(std::move(kw), nk));
          },
          "kmesh"_a, "gamma_centered"_a = true, "wrap_around"_a = false);

  nb::class_<gpu4dft::pw::KPointBasis>(pw_m, "KPointBasis")
      .def_ro("k_cart", &gpu4dft::pw::KPointBasis::k_cart)
      .def_ro("weight", &gpu4dft::pw::KPointBasis::weight)
      .def_ro("npw", &gpu4dft::pw::KPointBasis::npw)
      .def_prop_ro(
          "fft_indices",
          [](const gpu4dft::pw::KPointBasis& self) {
            return to_numpy_int_1d(self.fft_indices, self.fft_indices.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "kin",
          [](const gpu4dft::pw::KPointBasis& self) {
            return to_numpy_1d(self.kin, self.kin.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "kg_xyz",
          [](const gpu4dft::pw::KPointBasis& self) {
            return to_numpy_2d(self.kg_xyz, static_cast<std::size_t>(self.npw), 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "g_xyz",
          [](const gpu4dft::pw::KPointBasis& self) {
            return to_numpy_2d(self.g_xyz, static_cast<std::size_t>(self.npw), 3);
          },
          nb::rv_policy::move);

  nb::class_<gpu4dft::pw::PwBasis>(pw_m, "PwBasis")
      .def(
          "__init__",
          [](gpu4dft::pw::PwBasis* self, const gpu4dft::pw::PwCell& cell, double ecut_wfc,
             double ecut_rho, std::optional<ConstMat> kpoints, std::optional<ConstVec> kweights,
             std::array<int, 3> fft_grid_override) {
            std::vector<std::array<double, 3>> kpts_vec;
            std::vector<double> kw_vec;
            if (kpoints.has_value()) {
              if (kpoints->shape(1) != 3) {
                throw std::invalid_argument("PwBasis: kpoints must have shape (nk, 3)");
              }
              const std::size_t nk = kpoints->shape(0);
              const double* kp = kpoints->data();
              kpts_vec.resize(nk);
              for (std::size_t i = 0; i < nk; ++i) {
                kpts_vec[i] = {kp[3 * i + 0], kp[3 * i + 1], kp[3 * i + 2]};
              }
              if (kweights.has_value()) {
                if (kweights->shape(0) != nk) {
                  throw std::invalid_argument("PwBasis: kweights length must equal nk");
                }
                kw_vec.assign(kweights->data(), kweights->data() + nk);
              }
            }
            nb::gil_scoped_release release;
            new (self) gpu4dft::pw::PwBasis(cell, ecut_wfc, ecut_rho, kpts_vec, kw_vec,
                                            fft_grid_override);
          },
          "cell"_a, "ecut_wfc"_a, "ecut_rho"_a = 0.0, "kpoints"_a = nb::none(),
          "kweights"_a = nb::none(), "fft_grid_override"_a = std::array<int, 3>{0, 0, 0})
      .def_prop_ro("cell", &gpu4dft::pw::PwBasis::cell)
      .def_prop_ro("ecut_wfc", &gpu4dft::pw::PwBasis::ecut_wfc)
      .def_prop_ro("ecut_rho", &gpu4dft::pw::PwBasis::ecut_rho)
      .def_prop_ro("fft_grid", &gpu4dft::pw::PwBasis::fft_grid)
      .def_prop_ro("ngrid", &gpu4dft::pw::PwBasis::ngrid)
      .def_prop_ro("nkpts", &gpu4dft::pw::PwBasis::nkpts)
      .def("kpoint", &gpu4dft::pw::PwBasis::kpoint, "ik"_a = 0)
      .def_prop_ro("kpoints", &gpu4dft::pw::PwBasis::kpoints)
      .def_prop_ro(
          "g_vecs_xyz",
          [](const gpu4dft::pw::PwBasis& self) {
            return to_numpy_2d(self.g_vecs_xyz(), static_cast<std::size_t>(self.ngrid()), 3);
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "g2",
          [](const gpu4dft::pw::PwBasis& self) {
            return to_numpy_1d(self.g2(), static_cast<std::size_t>(self.ngrid()));
          },
          nb::rv_policy::move)
      .def("compute_local_pp_grid",
           [](const gpu4dft::pw::PwBasis& self) {
             std::vector<double> vloc;
             {
               nb::gil_scoped_release release;
               vloc = self.compute_local_pp_grid();
             }
             const std::size_t n = vloc.size();
             return to_numpy_1d(std::move(vloc), n);
           })
      .def(
          "compute_exx_coulomb_kernel",
          [](const gpu4dft::pw::PwBasis& self, std::array<double, 3> q_shift,
             gpu4dft::pw::ExxDiv exxdiv, std::array<int, 3> kmesh) {
            std::vector<double> vexx;
            {
              nb::gil_scoped_release release;
              vexx = self.compute_exx_coulomb_kernel(q_shift, exxdiv, kmesh);
            }
            const std::size_t n = vexx.size();
            return to_numpy_1d(std::move(vexx), n);
          },
          "q_shift"_a = std::array<double, 3>{0.0, 0.0, 0.0},
          "exxdiv"_a = gpu4dft::pw::ExxDiv::Ewald, "kmesh"_a = std::array<int, 3>{1, 1, 1});

  nb::class_<gpu4dft::pw::PwEigenResult>(pw_m, "PwEigenResult")
      .def_ro("iterations", &gpu4dft::pw::PwEigenResult::iterations)
      .def_ro("hpsi_calls", &gpu4dft::pw::PwEigenResult::hpsi_calls)
      .def_ro("max_residual", &gpu4dft::pw::PwEigenResult::max_residual)
      .def_prop_ro(
          "eigenvalues",
          [](const gpu4dft::pw::PwEigenResult& self) {
            return to_numpy_1d(self.eigenvalues, self.eigenvalues.size());
          },
          nb::rv_policy::move)
      .def_prop_ro(
          "eigenvectors",
          [](const gpu4dft::pw::PwEigenResult& self) {
            const std::size_t nb = self.eigenvalues.size();
            const std::size_t npw = (nb > 0) ? (self.eigenvectors.size() / nb) : 0;
            // Convert column-major [npw x nb] to row-major [npw x nb]
            std::vector<std::complex<double>> row_maj(npw * nb);
            for (std::size_t g = 0; g < npw; ++g) {
              for (std::size_t b = 0; b < nb; ++b) {
                row_maj[g * nb + b] = self.eigenvectors[b * npw + g];
              }
            }
            return to_numpy_complex_2d(std::move(row_maj), npw, nb);
          },
          nb::rv_policy::move);

  pw_m.def(
      "fft3d_batch",
      [](ConstComplex4D data, int direction, gpu4dft::Backend backend) {
        const int nb_dim = static_cast<int>(data.shape(0));
        const int n1 = static_cast<int>(data.shape(1));
        const int n2 = static_cast<int>(data.shape(2));
        const int n3 = static_cast<int>(data.shape(3));
        const std::size_t tot =
            static_cast<std::size_t>(nb_dim) * n1 * n2 * n3;
        const auto* in_ptr = data.data();
        std::vector<std::complex<double>> out(tot);
        {
          nb::gil_scoped_release release;
          gpu4dft::pw::fft3d_batch({n1, n2, n3}, nb_dim, in_ptr, out.data(), direction, backend);
        }
        return to_numpy_complex_4d(std::move(out), static_cast<std::size_t>(nb_dim),
                                   static_cast<std::size_t>(n1), static_cast<std::size_t>(n2),
                                   static_cast<std::size_t>(n3));
      },
      "data"_a, "direction"_a = -1, "backend"_a = gpu4dft::Backend::Cuda);

  pw_m.def(
      "solve_pw_eigenproblem",
      [](const gpu4dft::pw::PwBasis& basis, int ik, std::optional<ConstVec> vloc_r, int nband,
         gpu4dft::pw::PwSolverType solver, gpu4dft::Backend backend, double residual_tol,
         int max_iter, bool include_nonlocal_pp) {
        std::vector<double> zero_v;
        std::span<const double> v_span;
        if (vloc_r.has_value()) {
          if (static_cast<int>(vloc_r->shape(0)) != basis.ngrid()) {
            throw std::invalid_argument("solve_pw_eigenproblem: vloc_r length must equal ngrid");
          }
          v_span = std::span<const double>(vloc_r->data(), static_cast<std::size_t>(basis.ngrid()));
        } else {
          zero_v.assign(static_cast<std::size_t>(basis.ngrid()), 0.0);
          v_span = zero_v;
        }
        nb::gil_scoped_release release;
        return gpu4dft::pw::solve_pw_eigenproblem(basis, ik, v_span, nband, solver, backend,
                                                  residual_tol, max_iter, include_nonlocal_pp);
      },
      "basis"_a, "ik"_a = 0, "vloc_r"_a = nb::none(), "nband"_a = 4,
      "solver"_a = gpu4dft::pw::PwSolverType::Davidson, "backend"_a = gpu4dft::Backend::Cuda,
      "residual_tol"_a = 1e-10, "max_iter"_a = 80, "include_nonlocal_pp"_a = false);

  nb::class_<gpu4dft::pw::PwScfOptions>(pw_m, "PwScfOptions")
      .def(nb::init<>())
      .def_rw("backend", &gpu4dft::pw::PwScfOptions::backend)
      .def_rw("eigensolver", &gpu4dft::pw::PwScfOptions::eigensolver)
      .def_rw("xc", &gpu4dft::pw::PwScfOptions::xc)
      .def_rw("exxdiv", &gpu4dft::pw::PwScfOptions::exxdiv)
      .def_rw("kmesh", &gpu4dft::pw::PwScfOptions::kmesh)
      .def_rw("use_ace", &gpu4dft::pw::PwScfOptions::use_ace)
      .def_rw("charge", &gpu4dft::pw::PwScfOptions::charge)
      .def_rw("nbands", &gpu4dft::pw::PwScfOptions::nbands)
      .def_rw("max_scf_iter", &gpu4dft::pw::PwScfOptions::max_scf_iter)
      .def_rw("max_ace_iter", &gpu4dft::pw::PwScfOptions::max_ace_iter)
      .def_rw("max_eig_iter", &gpu4dft::pw::PwScfOptions::max_eig_iter)
      .def_rw("conv_tol", &gpu4dft::pw::PwScfOptions::conv_tol)
      .def_rw("density_tol", &gpu4dft::pw::PwScfOptions::density_tol)
      .def_rw("eig_tol", &gpu4dft::pw::PwScfOptions::eig_tol)
      .def_rw("mixing_beta", &gpu4dft::pw::PwScfOptions::mixing_beta)
      .def_rw("kerker_q0", &gpu4dft::pw::PwScfOptions::kerker_q0)
      .def_rw("diis_max_vec", &gpu4dft::pw::PwScfOptions::diis_max_vec)
      .def_rw("rho_cutoff", &gpu4dft::pw::PwScfOptions::rho_cutoff);

  nb::class_<gpu4dft::pw::PwScfResult>(pw_m, "PwScfResult")
      .def_ro("converged", &gpu4dft::pw::PwScfResult::converged)
      .def_ro("scf_iterations", &gpu4dft::pw::PwScfResult::scf_iterations)
      .def_ro("ace_iterations", &gpu4dft::pw::PwScfResult::ace_iterations)
      .def_ro("total_eig_iterations", &gpu4dft::pw::PwScfResult::total_eig_iterations)
      .def_ro("total_hpsi_calls", &gpu4dft::pw::PwScfResult::total_hpsi_calls)
      .def_ro("total_poisson_ffts", &gpu4dft::pw::PwScfResult::total_poisson_ffts)
      .def_ro("elapsed_ms", &gpu4dft::pw::PwScfResult::elapsed_ms)
      .def_ro("total_energy", &gpu4dft::pw::PwScfResult::total_energy)
      .def_ro("kinetic_energy", &gpu4dft::pw::PwScfResult::kinetic_energy)
      .def_ro("local_pp_energy", &gpu4dft::pw::PwScfResult::local_pp_energy)
      .def_ro("nonlocal_pp_energy", &gpu4dft::pw::PwScfResult::nonlocal_pp_energy)
      .def_ro("hartree_energy", &gpu4dft::pw::PwScfResult::hartree_energy)
      .def_ro("xc_energy", &gpu4dft::pw::PwScfResult::xc_energy)
      .def_ro("exx_energy", &gpu4dft::pw::PwScfResult::exx_energy)
      .def_ro("ewald_energy", &gpu4dft::pw::PwScfResult::ewald_energy)
      .def_ro("homo_energy", &gpu4dft::pw::PwScfResult::homo_energy)
      .def_ro("lumo_energy", &gpu4dft::pw::PwScfResult::lumo_energy)
      .def_ro("final_density_rms", &gpu4dft::pw::PwScfResult::final_density_rms)
      .def_ro("nbands", &gpu4dft::pw::PwScfResult::nbands)
      .def_ro("eigenvalues", &gpu4dft::pw::PwScfResult::eigenvalues)
      .def_prop_ro(
          "density_r",
          [](const gpu4dft::pw::PwScfResult& self) {
            return to_numpy_1d(self.density_r, self.density_r.size());
          },
          nb::rv_policy::move);

  nb::class_<gpu4dft::pw::PwKohnShamSolver>(pw_m, "PwKohnShamSolver")
      .def(
          "__init__",
          [](gpu4dft::pw::PwKohnShamSolver* self, const gpu4dft::pw::PwBasis& basis,
             const gpu4dft::pw::PwScfOptions& opts) {
            new (self) gpu4dft::pw::PwKohnShamSolver(basis, opts);
          },
          "basis"_a, "opts"_a = gpu4dft::pw::PwScfOptions{})
      .def("run", [](gpu4dft::pw::PwKohnShamSolver& self) {
        nb::gil_scoped_release release;
        return self.run();
      });

  pw_m.def(
      "ewald_energy",
      [](ConstMat a_bohr, const std::vector<int>& atomic_numbers, ConstMat coords_bohr,
         const std::string& pseudo_family, double precision) {
        gpu4dft::pw::PwCell cell(mat_to_lattice(a_bohr), make_pw_atoms(atomic_numbers, coords_bohr),
                                 pseudo_family);
        nb::gil_scoped_release release;
        return cell.ewald_energy(precision);
      },
      "a_bohr"_a, "atomic_numbers"_a, "coords_bohr"_a, "pseudo_family"_a = "gth-pade",
      "precision"_a = 1e-13);

  pw_m.def(
      "madelung_constant",
      [](ConstMat a_bohr, const std::vector<int>& atomic_numbers, ConstMat coords_bohr,
         const std::string& pseudo_family, std::array<int, 3> kmesh, double precision) {
        gpu4dft::pw::PwCell cell(mat_to_lattice(a_bohr), make_pw_atoms(atomic_numbers, coords_bohr),
                                 pseudo_family);
        nb::gil_scoped_release release;
        return cell.madelung_constant(kmesh, precision);
      },
      "a_bohr"_a, "atomic_numbers"_a, "coords_bohr"_a, "pseudo_family"_a = "gth-pade",
      "kmesh"_a = std::array<int, 3>{1, 1, 1}, "precision"_a = 1e-13);
}
