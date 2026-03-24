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

# Print all environment variables at the start of the script
echo "=== Environment Variables ==="
env | sort
echo "============================="
echo ""

SCRIPT_DIR=$(dirname "$(realpath "$0")")
LEGACY_TEST_PATH="${SCRIPT_DIR}/../Paddle/test/legacy_test"
PADDLE_DIR="${SCRIPT_DIR}/../Paddle"
PATCH_FILE="${SCRIPT_DIR}/../patches/test-corex.patch"
COREX_PATCH_APPLIED=0

apply_corex_patch() {
  if [[ ! -f "$PATCH_FILE" ]]; then
    echo "WARNING: $PATCH_FILE not found, skip CoreX patch."
    return 0
  fi
  echo "=== Applying $(basename "$PATCH_FILE") (under $PADDLE_DIR) ==="
  if (cd "$PADDLE_DIR" && patch -p1 --forward < "$PATCH_FILE"); then
    COREX_PATCH_APPLIED=1
    echo "Patch applied."
  else
    echo "ERROR: Failed to apply $PATCH_FILE" >&2
    return 1
  fi
}

revert_corex_patch() {
  if [[ "$COREX_PATCH_APPLIED" -ne 1 ]]; then
    return 0
  fi
  if [[ ! -f "$PATCH_FILE" ]]; then
    return 0
  fi
  echo "=== Reverting $(basename "$PATCH_FILE") ==="
  if (cd "$PADDLE_DIR" && patch -p1 -R < "$PATCH_FILE"); then
    echo "Patch reverted."
  else
    echo "WARNING: Failed to revert patch; check $PADDLE_DIR manually." >&2
  fi
  COREX_PATCH_APPLIED=0
}

trap revert_corex_patch EXIT

export LD_LIBRARY_PATH=/usr/local/corex/lib:${LD_LIBRARY_PATH}
export PYTHONPATH="${LEGACY_TEST_PATH}:${PYTHONPATH}"

python -m pip install parameterized

NUM_GPUS=$(ixsmi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPU found!"
    exit 1
fi
LAST_GPU=$((NUM_GPUS - 1))
echo "Using last GPU: $LAST_GPU"
export CUDA_VISIBLE_DEVICES=$LAST_GPU

export FLAG_SKIP_FLOAT64=1

apply_corex_patch || exit 1

CURRENT_DIR=$(pwd)
PADDLE_SOURCE_DIR="${CURRENT_DIR}/../Paddle"

mkdir -p build || { echo "ERROR: Failed to create build directory"; exit 1; }
cd build || { echo "ERROR: Failed to enter build directory"; exit 1; }

echo "=== Configuring project ==="
cmake .. || { echo "ERROR: CMake configuration failed"; exit 1; }

echo "=== Building project (if needed) ==="
make -j$(nproc) || { echo "ERROR: Build failed"; exit 1; }

echo "=== Running tests ==="
ctest --output-on-failure -V -j8 || { 
    echo "ERROR: Tests failed!" >&2
    echo "Exit code: $?" >&2
    exit 1
}

cd ../../

# Run legacy tests only if primary tests passed
echo "=== Running Paddle legacy tests ==="
# Navigate to unittest_runner directory
unset CUDA_VISIBLE_DEVICES
cd unittest_runner

# Run test_runner.py with specified parameters
python test_runner.py --path ../Paddle/test/legacy_test/ --skip-float64 --disabled-file ../tests/disabled_test.txt --rerun-failed

# Check if failed_logs directory has any log files
if [ -d "failed_logs" ] && [ "$(ls -A failed_logs)" ]; then
    echo "Failed tests found in CI. Displaying all failure logs:"
    for log_file in failed_logs/*; do
        if [ -f "$log_file" ]; then
            echo "=== $log_file ==="
            cat "$log_file"
            echo ""
        fi
    done
    echo "Legacy tests failed!"
    exit 1
else
    echo "All legacy tests passed in CI."
fi

echo "=== All tests passed successfully ==="

# Return to original directory
cd - > /dev/null
