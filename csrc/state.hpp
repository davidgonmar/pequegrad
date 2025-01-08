#pragma once
#include <string>

class GlobalState {
private:
  GlobalState() : cuda_allocator("default") {}

  static GlobalState *instance;

  std::string cuda_allocator;
  int cuda_stream;

public:
  GlobalState(const GlobalState &) = delete;
  GlobalState &operator=(const GlobalState &) = delete;

  static GlobalState *getInstance() {
    if (instance == nullptr) {
      instance = new GlobalState();
    }
    return instance;
  }

  std::string get_cuda_allocator() { return cuda_allocator; }

  void set_cuda_allocator(std::string allocator) { cuda_allocator = allocator; }

  int get_cuda_stream() { return cuda_stream; }

  void set_cuda_stream(int stream) { cuda_stream = stream; }
};
