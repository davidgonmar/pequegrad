#pragma once
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace pg {

namespace device {

enum class DeviceKind {
  CPU,
  CUDA,
};

constexpr DeviceKind CPU = DeviceKind::CPU;
constexpr DeviceKind CUDA = DeviceKind::CUDA;

inline std::string device_to_string(const DeviceKind device) {
  switch (device) {
  case DeviceKind::CPU:
    return "CPU";
  case DeviceKind::CUDA:
    return "CUDA";
  default:
    return "Unknown with integer value " +
           std::to_string(static_cast<int>(device));
  }
}

inline std::ostream &operator<<(std::ostream &os, const DeviceKind &device) {
  os << device_to_string(device);
  return os;
}

const DeviceKind &default_device();

std::shared_ptr<void> allocate(const size_t nbytes, const DeviceKind device,
                               bool pinned = false);

} // namespace device

} // namespace pg