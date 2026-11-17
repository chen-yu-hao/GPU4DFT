#pragma once

// GPU4DFT_HD marks functions that must compile for both host and device.
// Integral math lives in such functions so that the host reference backend
// runs exactly the arithmetic the GPU kernels run.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU4DFT_HD __host__ __device__
#else
#define GPU4DFT_HD
#endif
