#pragma once

#include <math.h>

#include "gpu4dft/config.hpp"

// Precision-overloaded math wrappers usable from host and device code.
// The global C functions are available on both sides under CUDA/HIP, so a
// single body serves every target.
namespace gpu4dft::math {

GPU4DFT_HD inline double sqrt(double x) { return ::sqrt(x); }
GPU4DFT_HD inline float sqrt(float x) { return ::sqrtf(x); }

GPU4DFT_HD inline double erf(double x) { return ::erf(x); }
GPU4DFT_HD inline float erf(float x) { return ::erff(x); }

GPU4DFT_HD inline double exp(double x) { return ::exp(x); }
GPU4DFT_HD inline float exp(float x) { return ::expf(x); }

}  // namespace gpu4dft::math
