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

echo "Cleaning build directories..."
rm -rf build
rm -rf Paddle/build

if [ -d "Paddle/.git" ]; then
    echo "Restoring Paddle git repository..."
    
    if [ -f ".paddle_original_head" ]; then
        ORIGINAL_HEAD=$(cat .paddle_original_head)
        git -C Paddle reset --hard "$ORIGINAL_HEAD"
        rm -f .paddle_original_head
        echo "Paddle successfully reset to original HEAD: $ORIGINAL_HEAD"
    else
        git -C Paddle reset --hard HEAD
        if git -C Paddle log -1 --pretty=%B | grep -q "^Revert "; then
            git -C Paddle reset --hard HEAD~1
            echo "Reverted the last commit (Fallback)."
        fi
    fi
    
    git -C Paddle clean -fd
    
    if [ -d "Paddle/third_party/warpctc/.git" ]; then
        git -C Paddle/third_party/warpctc reset --hard
    fi
    if [ -d "Paddle/third_party/eigen3/.git" ]; then
        git -C Paddle/third_party/eigen3 reset --hard
    fi
fi

echo "Clean up completed!"
