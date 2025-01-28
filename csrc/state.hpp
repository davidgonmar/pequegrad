#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// custom cuda stream class

class CudaStream {
public:
  explicit CudaStream(unsigned int flags = cudaStreamDefault) {
    if (cudaStreamCreateWithFlags(&stream_, flags) != cudaSuccess) {
      throw std::runtime_error("Failed to create CUDA stream");
    }
  }

  static std::shared_ptr<CudaStream> wrap_default() {
    return std::shared_ptr<CudaStream>(new CudaStream(0, true));
  }

  ~CudaStream() {
    if (stream_ && !is_default_) {
      cudaStreamDestroy(stream_);
    }
  }

  CudaStream(CudaStream &&other) noexcept
      : stream_(other.stream_), is_default_(other.is_default_) {
    other.stream_ = nullptr;
    other.is_default_ = false;
  }

  CudaStream &operator=(CudaStream &&other) noexcept {
    if (this != &other) {
      if (stream_ && !is_default_) {
        cudaStreamDestroy(stream_);
      }
      stream_ = other.stream_;
      is_default_ = other.is_default_;
      other.stream_ = nullptr;
      other.is_default_ = false;
    }
    return *this;
  }

  CudaStream(const CudaStream &) = delete;
  CudaStream &operator=(const CudaStream &) = delete;

  void synchronize() const {
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      throw std::runtime_error("Failed to synchronize CUDA stream");
    }
  }

  cudaStream_t get() const { return stream_; }

private:
  CudaStream(cudaStream_t stream, bool is_default)
      : stream_(stream), is_default_(is_default) {}

  cudaStream_t stream_ = nullptr;
  bool is_default_ = false;
};

class GlobalState {
private:
  GlobalState() : cuda_allocator("default") {}

  static GlobalState *instance;

  std::string cuda_allocator;
  std::shared_ptr<CudaStream> cuda_stream;

public:
  GlobalState(const GlobalState &) = delete;
  GlobalState &operator=(const GlobalState &) = delete;

  static GlobalState *getInstance() {
    if (instance == nullptr) {
      instance = new GlobalState();
      // default cuda stream
      instance->cuda_stream = std::make_shared<CudaStream>();
    }
    return instance;
  }

  std::string get_cuda_allocator() { return cuda_allocator; }

  void set_cuda_allocator(std::string allocator) { cuda_allocator = allocator; }

  std::shared_ptr<CudaStream> get_cuda_stream() { return cuda_stream; }

  void set_cuda_stream(std::shared_ptr<CudaStream> stream) {
    cuda_stream = stream;
  }
};
