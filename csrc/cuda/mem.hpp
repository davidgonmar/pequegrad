#pragma once
#include <memory>
#include <vector>

std::shared_ptr<void> allocate_cuda(const size_t nbytes);

void copy_from_cpu_to_cuda(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes);
void copy_from_cuda_to_cpu(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes);

void copy_from_cuda_to_cpu(const void *src, std::shared_ptr<void> &dst,
                           const size_t nbytes);
void copy_from_cpu_to_cuda(const void *src, std::shared_ptr<void> &dst,
                           const size_t nbytes);

void reset_custom_allocator();

std::vector<std::size_t> get_custom_allocator_alloc_history();