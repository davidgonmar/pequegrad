#pragma once
#include "cpu/mem.hpp"
#include "cuda/mem.hpp"
#include <cstdlib>
#include <iostream>
#include <map>
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

inline std::string device_kind_to_string(const DeviceKind device) {
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
  os << device_kind_to_string(device);
  return os;
}

std::shared_ptr<void> allocate(const size_t nbytes, const DeviceKind device,
                               bool pinned = false);

class Device {
protected:
  DeviceKind _kind;

public:
  ~Device() = default;

  virtual std::shared_ptr<void> allocate(const size_t nbytes,
                                         bool pinned = false) const = 0;

  virtual void memset(void *ptr, const int value,
                      const size_t nbytes) const = 0;

  virtual std::string str() const = 0;

  DeviceKind kind() const { return _kind; }
};
class CPUDevice : public Device {
  int _idx;

public:
  CPUDevice(int _idx) : _idx(_idx) { _kind = DeviceKind::CPU; }

  std::shared_ptr<void> allocate(const size_t nbytes,
                                 bool pinned = false) const override {
    return device::allocate(nbytes, this->_kind, pinned);
  }

  std::string str() const override { return "cpu:" + std::to_string(_idx); }

  void memset(void *ptr, const int value, const size_t nbytes) const override {
    throw std::runtime_error("Not implemented yet");
  }

  int idx() const { return _idx; }
};

class CudaDevice : public Device {
  int _idx;

public:
  CudaDevice(int _idx) : _idx(_idx) { _kind = DeviceKind::CUDA; }

  std::shared_ptr<void> allocate(const size_t nbytes,
                                 bool pinned = false) const override {
    return device::allocate(nbytes, this->_kind, pinned);
  }

  std::string str() const override { return "cuda:" + std::to_string(_idx); }

  void memset(void *ptr, const int value, const size_t nbytes) const override {
    throw std::runtime_error("Not implemented yet");
  }

  int idx() const { return _idx; }
};

// device registry singleton
class DeviceRegistry {
  std::map<std::string, std::shared_ptr<Device>> _devices;

  DeviceRegistry() {
    _devices["cpu:0"] = std::make_shared<CPUDevice>(0);
    _devices["cuda:0"] = std::make_shared<CudaDevice>(0);
  }

public:
  DeviceRegistry(const DeviceRegistry &) = delete;
  DeviceRegistry &operator=(const DeviceRegistry &) = delete;

  static DeviceRegistry &get_instance() {
    static DeviceRegistry instance;
    return instance;
  }

  std::shared_ptr<Device> get(std::string key) {
    // if it is cpu or cuda alone, return 0
    if (key == "cpu" || key == "cuda") {
      key += ":0";
    }
    if (_devices.find(key) == _devices.end()) {
      throw std::runtime_error("Device not found: " + key);
    }
    return _devices.at(key);
  }

  std::shared_ptr<Device> get(DeviceKind kind, int idx = 0) {
    if (kind == DeviceKind::CPU) {
      return _devices.at("cpu:0");
    } else if (kind == DeviceKind::CUDA) {
      return _devices.at("cuda:0");
    }
  }
};

static std::shared_ptr<Device> get_default_device() {
  return DeviceRegistry::get_instance().get("cpu");
}

static inline std::string
device_to_string(const std::shared_ptr<Device> device) {
  return device->str();
}

static inline std::ostream &operator<<(std::ostream &os,
                                       const std::shared_ptr<Device> &device) {
  os << device_to_string(device);
  return os;
}

// equal operator for both device and shared_ptr<device>
static inline bool operator==(const std::shared_ptr<Device> &lhs,
                              const std::shared_ptr<Device> &rhs) {
  return lhs->kind() == rhs->kind();
}

static bool is_cuda(const std::shared_ptr<Device> &device) {
  return device->kind() == DeviceKind::CUDA;
}

static bool is_cpu(const std::shared_ptr<Device> &device) {
  return device->kind() == DeviceKind::CPU;
}

std::shared_ptr<Device> from_str(std::string str);

std::shared_ptr<Device> from_kind(DeviceKind kind);

} // namespace device

} // namespace pg