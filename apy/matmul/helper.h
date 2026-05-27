// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ap {

template <typename Element>
__global__ void PackBRowMajorToColumnMajorKernel(const Element* src,
                                                 Element* dst,
                                                 int64_t k_size,
                                                 int64_t n_size,
                                                 int64_t batch_count,
                                                 int64_t src_batch_stride) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = batch_count * k_size * n_size;
  if (idx >= total) {
    return;
  }

  int64_t n = idx % n_size;
  int64_t k = (idx / n_size) % k_size;
  int64_t batch = idx / (k_size * n_size);
  int64_t src_base = src_batch_stride == 0 ? 0 : batch * src_batch_stride;

  // src is Paddle's row-major [K, N]. dst is packed as [N, K] for the
  // existing IX11 TN SME mainloop.
  dst[batch * n_size * k_size + n * k_size + k] =
      src[src_base + k * n_size + n];
}

template <typename Element>
cudaError_t LaunchPackBRowMajorToColumnMajor(const Element* src,
                                             Element* dst,
                                             int64_t k_size,
                                             int64_t n_size,
                                             int64_t batch_count,
                                             int64_t src_batch_stride,
                                             cudaStream_t stream) {
  int64_t total = batch_count * k_size * n_size;
  if (total == 0) {
    return cudaSuccess;
  }

  constexpr int kThreads = 256;
  int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  PackBRowMajorToColumnMajorKernel<Element>
      <<<blocks, kThreads, 0, stream>>>(
          src, dst, k_size, n_size, batch_count, src_batch_stride);
  return cudaGetLastError();
}

}  // namespace ap
