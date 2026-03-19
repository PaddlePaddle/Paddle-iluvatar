#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

PYTHON_VERSION=${PYTHON_VERSION:-$(python3 -V 2>&1|awk '{print $2}')}
COREX_VERSION=${COREX_VERSION:-latest}
if [[ "${COREX_VERSION}" == "latest" ]]; then
  COREX_VERSION=`date --utc +%Y%m%d%H%M%S`
fi
BUILD_TEST=${BUILD_TEST:-1}
COREX_ARCH=${COREX_ARCH:-ivcore11}
export CMAKE_CUDA_ARCHITECTURES=${COREX_ARCH}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PADDLE_SOURCE_DIR="${SCRIPT_DIR}/Paddle"
PADDLE_BUILD_DIR="${PADDLE_SOURCE_DIR}/build"
ILUVATAR_SOURCE_DIR="${SCRIPT_DIR}"
ILUVATAR_BUILD_DIR="${PADDLE_BUILD_DIR}/custom_device_build"
PATCH_FILE="${SCRIPT_DIR}/patches/paddle-corex.patch"
STATE_FILE="${SCRIPT_DIR}/.build_state"

BUILD_WITH_FLAGCX=0
FLAGCX_ROOT="/workspace/FlagCX"
PLATFORM_ID=$(uname -i)

if [ "$BUILD_WITH_FLAGCX" == "1" ]; then
  WITH_FLAGCX="ON"
else
  WITH_FLAGCX="OFF"
fi

if [[ "$1" == "--clean" ]]; then
  echo "Cleaning build environment..."
  if [[ -d "$PADDLE_BUILD_DIR" ]]; then
    rm -rf "$PADDLE_BUILD_DIR"
    echo "Removed build directory"
  fi
  if [[ -d "${PADDLE_SOURCE_DIR}" && -f "${PATCH_FILE}" ]]; then
    if git -C "${PADDLE_SOURCE_DIR}" apply --reverse --check "${PATCH_FILE}" &>/dev/null; then
      git -C "${PADDLE_SOURCE_DIR}" apply --reverse "${PATCH_FILE}" && echo "Patch reverted" || true
    fi
  fi
  _warpctc="${PADDLE_SOURCE_DIR}/third_party/warpctc"
  if [[ -d "${_warpctc}/.git" ]] || git -C "${_warpctc}" rev-parse --is-inside-work-tree &>/dev/null; then
    git -C "${_warpctc}" reset --hard &>/dev/null && echo "Restored Paddle/third_party/warpctc" || true
  fi
  _eigen="${PADDLE_SOURCE_DIR}/third_party/eigen3"
  if [[ -d "${_eigen}/.git" ]]; then
    git -C "${_eigen}" reset --hard &>/dev/null && echo "eigen reset" || true
  fi
  [[ -f "$STATE_FILE" ]] && rm -f "$STATE_FILE" && echo "Removed state file"
  echo "Clean completed!"
  exit 0
fi

if [[ ! -f "$STATE_FILE" ]]; then
  echo "First time build detected. Setting up environment..."
  if ! git -C "$PADDLE_SOURCE_DIR" apply --reverse --check "$PATCH_FILE" > /dev/null 2>&1; then
    if ! git -C "$PADDLE_SOURCE_DIR" apply "$PATCH_FILE"; then
      echo "Error: Failed to apply patch!"
      exit 1
    fi
    echo "Patch applied successfully!"
  fi
  cp -r "${SCRIPT_DIR}/patches/eigen/Core" "${PADDLE_SOURCE_DIR}/third_party/eigen3/Eigen/Core"
  cp -r "${SCRIPT_DIR}/patches/eigen/Tensor" "${PADDLE_SOURCE_DIR}/third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
  cp -r "${SCRIPT_DIR}/patches/eigen/TensorAssign.h" "${PADDLE_SOURCE_DIR}/third_party/eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorAssign.h"
  echo "BUILD_ENV_SET=1" > "$STATE_FILE"
  echo "Environment setup completed"
else
  echo "Incremental build detected. Skipping environment setup."
fi

if [[ ! -d "${PADDLE_BUILD_DIR}" ]]; then
  mkdir -p "${PADDLE_BUILD_DIR}"
fi
if [[ ! -d "${ILUVATAR_BUILD_DIR}" ]]; then
  mkdir -p "${ILUVATAR_BUILD_DIR}"
fi

PADDLE_CMAKE_ARGS=(
  "-DPY_VERSION=${PYTHON_VERSION}"
  "-DWITH_GPU=OFF"
  "-DWITH_DISTRIBUTE=ON"
  "-DWITH_CUSTOM_DEVICE_SUB_BUILD=ON"
  "-DCUSTOM_DEVICE_SOURCE_DIR=${ILUVATAR_SOURCE_DIR}"
)
CUSTOM_DEVICE_CMAKE_ARGS=(
  "-DWITH_COREX=ON"
  "-DPADDLE_SOURCE_DIR=${PADDLE_SOURCE_DIR}"
  "-DWITH_NCCL=ON"
  "-DNCCL_VERSION=0"
  "-DWITH_FLAGCX=${WITH_FLAGCX}"
  "-DWITH_RCCL=OFF"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
  "-DON_INFER=ON"
  "-DCOREX_VERSION=${COREX_VERSION}"
  "-DCOREX_ARCH=${COREX_ARCH}"
  "-DFLAGCX_ROOT=${FLAGCX_ROOT}"
  "-DCMAKE_CXX_FLAGS=-Wno-error=pessimizing-move -Wno-error=deprecated-copy -Wno-error=init-list-lifetime -pthread"
  "-DCMAKE_CUDA_FLAGS=-Xclang -fcuda-allow-variadic-functions -mllvm --skip-double"
  "-DCMAKE_C_FLAGS=-pthread"
  "-DWITH_DGC=OFF"
)
if [[ "${PLATFORM_ID}" == "aarch64" ]]; then
  CUSTOM_DEVICE_CMAKE_ARGS+=("-DWITH_ARM=ON")
else
  CUSTOM_DEVICE_CMAKE_ARGS+=("-DWITH_ARM=OFF")
fi
CUSTOM_DEVICE_CMAKE_ARGS_STR=$(IFS=';'; echo "${CUSTOM_DEVICE_CMAKE_ARGS[*]}")
PADDLE_CMAKE_ARGS+=("-DCUSTOM_DEVICE_CMAKE_ARGS=${CUSTOM_DEVICE_CMAKE_ARGS_STR}")

pushd "${PADDLE_BUILD_DIR}"
if [[ ! -f "build.ninja" ]]; then
  cmake -G Ninja "${PADDLE_CMAKE_ARGS[@]}" "${PADDLE_SOURCE_DIR}" \
    || { echo "Error: CMake configuration failed!"; exit 1; }
fi
if [[ "${PLATFORM_ID}" == "aarch64" ]]; then
  env TARGET=ARMV8 ninja -j$(nproc) 2>&1 | tee compile.log
else
  ninja -j$(nproc) 2>&1 | tee compile.log
fi
[[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "Error: Paddle build failed!"; exit 1; }
popd

PKG_DIR="${PADDLE_SOURCE_DIR}/build/python/dist"
PKG_NAME="paddlepaddle_iluvatar"
latest_pkg="$(ls -t "${PKG_DIR}" 2>/dev/null | grep "${PKG_NAME}" | head -1)"
if [[ -z "${latest_pkg}" ]]; then
  echo "ERROR: No ${PKG_NAME} package found in ${PKG_DIR}"
  exit 1
fi
PYTHON_PATH=$(which python3)
echo "Uninstalling old paddlepaddle-iluvatar..."
${PYTHON_PATH} -m pip uninstall paddlepaddle-iluvatar -y 2>/dev/null || true
echo "Installing ${latest_pkg}..."
${PYTHON_PATH} -m pip install "${PKG_DIR}/${latest_pkg}" || exit 1

echo "Build and installation completed successfully!"
exit 0
