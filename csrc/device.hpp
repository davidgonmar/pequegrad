#pragma once
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace pg {

namespace device {

enum class DeviceKind {
  CPU,
  CUDA,
};

constexpr DeviceKind CPU = DeviceKind::CPU;
constexpr DeviceKind CUDA = DeviceKind::CUDA;

const DeviceKind &default_device();

std::shared_ptr<void> allocate(const size_t nbytes, const DeviceKind device);

} // namespace device

} // namespace pg