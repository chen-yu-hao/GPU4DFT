#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/eri_cuda.hpp"

namespace gpu4dft::detail {
namespace {

void check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

// Owning device buffer. Placeholder for the pooled allocator the SCF driver
// will need; per-call cudaMalloc is fine for validation-sized problems.
template <typename T>
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t size) : size_(size) {
    if (size_ > 0) check(cudaMalloc(&ptr_, size_ * sizeof(T)), "cudaMalloc");
  }
  explicit DeviceArray(const std::vector<T>& host) : DeviceArray(host.size()) {
    if (size_ > 0) {
      check(cudaMemcpy(ptr_, host.data(), size_ * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy (host to device)");
    }
  }
  ~DeviceArray() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  T* data() const { return ptr_; }

  void download(std::vector<T>& host) const {
    if (host.size() != size_) throw std::logic_error("DeviceArray::download: size mismatch");
    if (size_ > 0) {
      check(cudaMemcpy(host.data(), ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost),
            "cudaMemcpy (device to host)");
    }
  }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

class DeviceShellPairs {
 public:
  explicit DeviceShellPairs(const ShellPairData& pairs)
      : npair_(pairs.npair()),
        prim_offset_(pairs.prim_offset),
        exponent_(pairs.exponent),
        Px_(pairs.Px),
        Py_(pairs.Py),
        Pz_(pairs.Pz),
        prefactor_(pairs.prefactor) {}

  ShellPairView view() const {
    return ShellPairView{npair_,       prim_offset_.data(), exponent_.data(),  Px_.data(),
                         Py_.data(),   Pz_.data(),          prefactor_.data()};
  }

 private:
  int npair_;
  DeviceArray<int> prim_offset_;
  DeviceArray<double> exponent_;
  DeviceArray<double> Px_;
  DeviceArray<double> Py_;
  DeviceArray<double> Pz_;
  DeviceArray<double> prefactor_;
};

}  // namespace

int cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();  // clear the sticky error
    return 0;
  }
  return count;
}

void run_eri_kernel_cuda(EriLaunchFn launch, const ShellPairData& bra, const ShellPairData& ket,
                         std::vector<double>& out) {
  const DeviceShellPairs device_bra(bra);
  const DeviceShellPairs device_ket(ket);
  DeviceArray<double> device_out(out.size());
  launch(device_bra.view(), device_ket.view(), device_out.data(), nullptr);
  check(cudaGetLastError(), "ERI kernel launch");
  check(cudaDeviceSynchronize(), "ERI kernel execution");
  device_out.download(out);
}

}  // namespace gpu4dft::detail
