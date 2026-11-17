#pragma once

#include <stdexcept>
#include <string>

namespace gpu4dft {

enum class Backend {
  Host,  // CPU reference backend, always built
  Cuda,  // NVIDIA GPUs, built when a CUDA compiler is available
};

// Precision used to evaluate integrals. Sums over primitives are always
// accumulated in double.
enum class Precision {
  FP64,
  FP32,
};

// Raised for functionality that is planned but not implemented yet (for
// example integral classes beyond (ss|ss)).
class NotImplemented : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

const char* to_string(Backend backend);
const char* to_string(Precision precision);

}  // namespace gpu4dft
