#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/eri.hpp"
#include "gpu4dft/kernel_registry.hpp"
#include "gpu4dft/types.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Array4 = nb::ndarray<nb::numpy, double, nb::ndim<4>>;

// Hands ownership of `data` to a NumPy array without copying.
Array4 to_numpy(std::vector<double> data, std::size_t n) {
  auto* owned = new std::vector<double>(std::move(data));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return Array4(owned->data(), {n, n, n, n}, owner);
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "GPU4DFT native core";

  nb::exception<gpu4dft::NotImplemented>(m, "NotImplementedError", PyExc_NotImplementedError);

  nb::enum_<gpu4dft::Backend>(m, "Backend")
      .value("host", gpu4dft::Backend::Host)
      .value("cuda", gpu4dft::Backend::Cuda);

  nb::enum_<gpu4dft::Precision>(m, "Precision")
      .value("fp64", gpu4dft::Precision::FP64)
      .value("fp32", gpu4dft::Precision::FP32);

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

  m.def("backend_available", &gpu4dft::backend_available, "backend"_a);

  m.def(
      "eri_tensor",
      [](const gpu4dft::Basis& basis, gpu4dft::Backend backend, gpu4dft::Precision precision) {
        std::vector<double> eri;
        {
          nb::gil_scoped_release release;
          eri = gpu4dft::eri_tensor(basis, backend, precision);
        }
        return to_numpy(std::move(eri), static_cast<std::size_t>(basis.nao()));
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
}
