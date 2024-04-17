#include "device.hpp"
#include <memory>
#include <stdexcept>
#include "cpu/mem.hpp"
#include "cuda/mem.hpp"

namespace pg {
namespace device {
static DeviceKind _default_device = DeviceKind::CPU;

const DeviceKind &default_device() { return _default_device; }

std::shared_ptr<void> allocate(const size_t nbytes, const DeviceKind device) {
  if (device == DeviceKind::CPU) {
    return allocate_cpu(nbytes);
  }
  else if (device == DeviceKind::CUDA) {
    return allocate_cuda(nbytes);
  }
  else {
    throw std::runtime_error("Unsupported device");
  }
}

}
} // namespace pg