#pragma once

namespace pg {

enum class Device {
  CPU,
  CUDA,
};

const Device &default_device();

} // namespace pg