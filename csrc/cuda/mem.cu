#include "cuda_utils.cuh"
#include "mem.hpp"

#include "state.hpp"
#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>

class CustomAllocator {
private:
  struct Block {
    void *ptr;
    size_t size;

    bool operator<(const Block &other) const { return ptr < other.ptr; }
  };

  void *base_memory;
  size_t total_size;
  std::vector<Block> free_list;
  const size_t alignment = 256; // Default CUDA alignment
  std::mutex alloc_mutex;       // Mutex for thread safety

  // Align a size up to the nearest multiple of `alignment`
  size_t align_size(size_t size) const {
    return (size + alignment - 1) & ~(alignment - 1);
  }

  std::vector<Block> allocated;

public:
  std::vector<std::size_t> alloc_history;
  CustomAllocator(size_t size) {
    total_size = align_size(size);
    if (cudaMalloc(&base_memory, total_size) != cudaSuccess) {
      throw std::runtime_error("Failed to allocate CUDA memory!");
    }
    std::cout << "Base memory allocated at address " << base_memory
              << std::endl;
    // let base mem be a multiple of 256
    free_list.push_back({base_memory, total_size});
  }

  ~CustomAllocator() {
    std::cout << "Deallocating base memory at address " << base_memory
              << std::endl;
    cudaFree(base_memory);
  }

  void *allocate(size_t size) {
    std::lock_guard<std::mutex> lock(alloc_mutex); // Ensure thread safety
    size_t aligned_size = align_size(size);

    // Find the best-fit block
    auto best_fit = free_list.end();
    for (auto it = free_list.begin(); it != free_list.end(); ++it) {
      if (it->size >= aligned_size &&
          (best_fit == free_list.end() || it->size < best_fit->size)) {
        best_fit = it;
      }
    }

    if (best_fit == free_list.end()) {
      std::string msg =
          "Out of memory! No suitable block found for an allocation of size " +
          std::to_string(size) + " and aligned size " +
          std::to_string(aligned_size);
      // print also stats (total allocated memory, total free memory, number of
      // blocks)
      msg += "\nTotal allocated memory: " + std::to_string(total_size) +
             " bytes\n";
      size_t total_free_memory = 0;
      for (const auto &block : free_list) {
        total_free_memory += block.size;
      }
      msg += "Total free memory: " + std::to_string(total_free_memory) +
             " bytes\n";
      msg +=
          "Number of free blocks: " + std::to_string(free_list.size()) + "\n";

      throw std::runtime_error(msg);
    }
    // Allocate from the best-fit block
    void *allocated_ptr = best_fit->ptr;

    if (best_fit->size > aligned_size) {
      // Split the block
      best_fit->ptr = static_cast<char *>(best_fit->ptr) + aligned_size;
      best_fit->size -= aligned_size;
    } else {
      // Fully consume the block
      free_list.erase(best_fit);
    }
    this->alloc_history.push_back((std::size_t)allocated_ptr);
    allocated.push_back({allocated_ptr, aligned_size});
    return allocated_ptr;
  }

  void deallocate(void *ptr, size_t size) {
    std::lock_guard<std::mutex> lock(alloc_mutex); // Ensure thread safety
    bool found = false;
    for (size_t i = 0; i < allocated.size(); i++) {
      if (allocated[i].ptr == ptr && align_size(size) == allocated[i].size) {
        free_list.push_back({ptr, allocated[i].size});
        allocated.erase(allocated.begin() + i);
        found = true;
        break;
      }
    }
    merge_free_blocks();
  }

  void reset() {
    std::lock_guard<std::mutex> lock(alloc_mutex);
    free_list.clear();
    free_list.push_back({base_memory, total_size});
    allocated.clear();
    this->alloc_history.clear();
  }

private:
  void merge_free_blocks() {
    std::sort(free_list.begin(), free_list.end());

    for (size_t i = 0; i < free_list.size() - 1;) {
      auto &current = free_list[i];
      auto &next = free_list[i + 1];

      // Check if the current and next blocks are contiguous
      if (static_cast<char *>(current.ptr) + current.size == next.ptr) {
        current.size += next.size;
        free_list.erase(free_list.begin() + i + 1);
      } else {
        ++i;
      }
    }
  }
};

class AllocatorWrapper {
private:
  static AllocatorWrapper *instance;
  CustomAllocator *allocator;

  AllocatorWrapper() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t allocation_size = free_mem * 0.4; // Allocate 60% of the free memory

    std::cout << "Allocating " << allocation_size / (1024 * 1024)
              << " MB for the custom allocator.\n";

    allocator = new CustomAllocator(allocation_size);
  }

public:
  AllocatorWrapper(const AllocatorWrapper &) = delete;
  AllocatorWrapper &operator=(const AllocatorWrapper &) = delete;

  ~AllocatorWrapper() { delete allocator; }

  static AllocatorWrapper *getInstance() {
    if (!instance) {
      instance = new AllocatorWrapper();
    }
    return instance;
  }

  CustomAllocator *getAllocator() { return allocator; }

  static void destroyInstance() {
    delete instance;
    instance = nullptr;
  }
};

AllocatorWrapper *AllocatorWrapper::instance = nullptr;

std::shared_ptr<void> allocate_cuda_default(const size_t nbytes) {
  void *ptr;
  CHECK_CUDA(cudaMallocAsync(&ptr, nbytes, 0));
  return std::shared_ptr<void>(
      ptr, [](void *p) { CHECK_CUDA(cudaFreeAsync(p, 0)); });
}

void copy_from_cpu_to_cuda_default(const std::shared_ptr<void> &src,
                                   const std::shared_ptr<void> &dst,
                                   const size_t nbytes) {
  cudaMemcpyAsync(dst.get(), src.get(), nbytes, cudaMemcpyHostToDevice, 0);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cpu_to_cuda_default(const void *src, std::shared_ptr<void> &dst,
                                   const size_t nbytes) {
  cudaMemcpyAsync(dst.get(), src, nbytes, cudaMemcpyHostToDevice, 0);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cuda_to_cpu_default(const std::shared_ptr<void> &src,
                                   const std::shared_ptr<void> &dst,
                                   const size_t nbytes) {
  cudaMemcpyAsync(dst.get(), src.get(), nbytes, cudaMemcpyDeviceToHost, 0);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cuda_to_cpu_default(const void *src, std::shared_ptr<void> &dst,
                                   const size_t nbytes) {
  cudaMemcpyAsync(dst.get(), src, nbytes, cudaMemcpyDeviceToHost, 0);
  CHECK_CUDA(cudaGetLastError());
}

std::shared_ptr<void> allocate_cuda_custom(const size_t nbytes) {
  CustomAllocator *allocator = AllocatorWrapper::getInstance()->getAllocator();
  void *ptr = allocator->allocate(nbytes);
  return std::shared_ptr<void>(
      ptr, [allocator, nbytes](void *p) { allocator->deallocate(p, nbytes); });
}

void copy_from_cpu_to_cuda_custom(const std::shared_ptr<void> &src,
                                  const std::shared_ptr<void> &dst,
                                  const size_t nbytes) {
  cudaMemcpyAsync(dst.get(), src.get(), nbytes, cudaMemcpyHostToDevice, 0);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cpu_to_cuda_custom(const void *src, std::shared_ptr<void> &dst,
                                  const size_t nbytes) {
  cudaMemcpyAsync(dst.get(), src, nbytes, cudaMemcpyHostToDevice, 0);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cuda_to_cpu_custom(const std::shared_ptr<void> &src,
                                  const std::shared_ptr<void> &dst,
                                  const size_t nbytes) {
  cudaMemcpyAsync(dst.get(), src.get(), nbytes, cudaMemcpyDeviceToHost, 0);
  CHECK_CUDA(cudaGetLastError());
}

void copy_from_cuda_to_cpu_custom(const void *src, std::shared_ptr<void> &dst,
                                  const size_t nbytes) {
  cudaMemcpyAsync(dst.get(), src, nbytes, cudaMemcpyDeviceToHost, 0);
  CHECK_CUDA(cudaGetLastError());
}

std::shared_ptr<void> allocate_cuda(const size_t nbytes) {
  if (GlobalState::getInstance()->get_cuda_allocator() == "default") {
    return allocate_cuda_default(nbytes);
  } else if (GlobalState::getInstance()->get_cuda_allocator() == "custom") {
    return allocate_cuda_custom(nbytes);
  } else {
    throw std::runtime_error("Unknown CUDA allocator: " +
                             GlobalState::getInstance()->get_cuda_allocator());
  }
}

void copy_from_cpu_to_cuda(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  if (GlobalState::getInstance()->get_cuda_allocator() == "default") {
    copy_from_cpu_to_cuda_default(src, dst, nbytes);
  } else if (GlobalState::getInstance()->get_cuda_allocator() == "custom") {
    copy_from_cpu_to_cuda_custom(src, dst, nbytes);
  } else {
    throw std::runtime_error("Unknown CUDA allocator: " +
                             GlobalState::getInstance()->get_cuda_allocator());
  }
}

void copy_from_cuda_to_cpu(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  if (GlobalState::getInstance()->get_cuda_allocator() == "default") {
    copy_from_cuda_to_cpu_default(src, dst, nbytes);
  } else if (GlobalState::getInstance()->get_cuda_allocator() == "custom") {
    copy_from_cuda_to_cpu_custom(src, dst, nbytes);
  } else {
    throw std::runtime_error("Unknown CUDA allocator: " +
                             GlobalState::getInstance()->get_cuda_allocator());
  }
}

void copy_from_cuda_to_cpu(const void *src, std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  if (GlobalState::getInstance()->get_cuda_allocator() == "default") {
    copy_from_cuda_to_cpu_default(src, dst, nbytes);
  } else if (GlobalState::getInstance()->get_cuda_allocator() == "custom") {
    copy_from_cuda_to_cpu_custom(src, dst, nbytes);
  } else {
    throw std::runtime_error("Unknown CUDA allocator: " +
                             GlobalState::getInstance()->get_cuda_allocator());
  }
}

void copy_from_cpu_to_cuda(const void *src, std::shared_ptr<void> &dst,
                           const size_t nbytes) {
  if (GlobalState::getInstance()->get_cuda_allocator() == "default") {
    copy_from_cpu_to_cuda_default(src, dst, nbytes);
  } else if (GlobalState::getInstance()->get_cuda_allocator() == "custom") {
    copy_from_cpu_to_cuda_custom(src, dst, nbytes);
  } else {
    throw std::runtime_error("Unknown CUDA allocator: " +
                             GlobalState::getInstance()->get_cuda_allocator());
  }
}

void reset_custom_allocator() {
  AllocatorWrapper::getInstance()->getAllocator()->reset();
}

std::vector<std::size_t> get_custom_allocator_alloc_history() {
  return AllocatorWrapper::getInstance()->getAllocator()->alloc_history;
}