
// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "cinn_interface.h"  // NOLINT

#include <cstring>  // For memset
#include <iostream>

namespace paddle {
namespace custom_device {
namespace iluvatar {

extern C_Status IluvatarCompile(void* dev_ptr,
                                const char* code,
                                char* out_path,
                                size_t len);

extern const char* IluvatarGetRuntimeSource(void* dev_ptr);

extern C_Status IluvatarModuleLoad(void* dev_ptr,
                                   const char* path,
                                   void** mod_out);

extern C_Status IluvatarModuleUnload(void* dev_ptr, void* module_handle);

extern C_Status IluvatarGetKernelAddress(void* dev_ptr,
                                         void* module_handle,
                                         const char* func_name,
                                         void** func_out);

extern C_Status IluvatarLaunchKernel(void* dev_ptr,
                                     void* func_ptr,
                                     void** args,
                                     int num_args,
                                     int gx,
                                     int gy,
                                     int gz,
                                     int bx,
                                     int by,
                                     int bz,
                                     int shm,
                                     void* stream);

extern C_Status IluvatarApplyCustomPass(void* dev_ptr, void* ir_module);

static C_CinnInterface iluvatar_cinn_impl;

void InitCinnInterface(C_DeviceInterface* device_interface) {
  std::memset(&iluvatar_cinn_impl, 0, sizeof(C_CinnInterface));

  iluvatar_cinn_impl.size = sizeof(C_CinnInterface);

  iluvatar_cinn_impl.dev_ptr = nullptr;

  iluvatar_cinn_impl.compile = IluvatarCompile;
  iluvatar_cinn_impl.get_runtime_source = IluvatarGetRuntimeSource;

  iluvatar_cinn_impl.module_load = IluvatarModuleLoad;
  iluvatar_cinn_impl.module_unload = IluvatarModuleUnload;
  iluvatar_cinn_impl.get_kernel_address = IluvatarGetKernelAddress;
  iluvatar_cinn_impl.launch_kernel = IluvatarLaunchKernel;

  iluvatar_cinn_impl.apply_custom_pass = IluvatarApplyCustomPass;

  if (device_interface) {
    device_interface->cinn_interface = &iluvatar_cinn_impl;
  } else {
    std::cerr << "[Iluvatar] Error: device_interface is null during CINN init."
              << std::endl;
  }
}

}  // namespace iluvatar
}  // namespace custom_device
}  // namespace paddle
