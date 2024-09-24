#include "device.hpp"
#include "cpu/mem.hpp"
#include "cuda/mem.hpp"
#include <memory>
#include <stdexcept>

namespace pg {
namespace device {
static DeviceKind _default_device = DeviceKind::CPU;

const DeviceKind &default_device() { return _default_device; }

std::shared_ptr<void> allocate(const size_t nbytes, const DeviceKind device,
                               bool pinned) {
  if (device == DeviceKind::CPU) {
    return allocate_cpu(nbytes, pinned);
  } else if (device == DeviceKind::CUDA) {
    return allocate_cuda(nbytes);
  } else {
    throw std::runtime_error("Unsupported device");
  }
}

std::shared_ptr<Device> from_str(std::string str) {
  return DeviceRegistry::get_instance().get(str);
}

std::shared_ptr<Device> from_kind(DeviceKind kind) {
  return DeviceRegistry::get_instance().get(kind);
}

} // namespace device
} // namespace pg