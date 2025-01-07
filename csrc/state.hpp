#pragma once
#include <string>

class GlobalState {
private:
  GlobalState() : cuda_allocator("default") {}

  static GlobalState *instance;

  std::string cuda_allocator;

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
};
