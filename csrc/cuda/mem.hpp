#pragma once
#include <memory>

std::shared_ptr<void> allocate_cuda(const size_t nbytes);

void copy_from_cpu_to_cuda(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes);
void copy_from_cuda_to_cpu(const std::shared_ptr<void> &src,
                           const std::shared_ptr<void> &dst,
                           const size_t nbytes);