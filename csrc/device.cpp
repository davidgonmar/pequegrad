#include "device.hpp"

namespace pg {

static Device _default_device = Device::CPU;

const Device &default_device() { return _default_device; }
} // namespace pg