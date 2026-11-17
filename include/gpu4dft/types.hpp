#pragma once

#include <stdexcept>
#include <string>

namespace gpu4dft {

enum class Backend {
  Host,  // CPU reference backend, always built
  Cuda,  // NVIDIA GPUs, built when a CUDA compiler is available
  Hip,   // AMD ROCm GPUs, built when a HIP compiler is available
};

// Precision used to evaluate integrals. Sums over primitives are always
// accumulated in double.
enum class Precision {
  FP64,
  FP32,
  Mixed,  // Dynamic per-quartet FP64 / FP32 dispatch based on Schwarz*density bound
};

// Raised for functionality that is planned but not implemented yet.
class NotImplemented : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

const char* to_string(Backend backend);
const char* to_string(Precision precision);

}  // namespace gpu4dft
