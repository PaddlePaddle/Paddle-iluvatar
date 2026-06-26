#!/usr/bin/env bash
set -e

export FLAGS_prim_all=True
export FLAGS_prim_enable_dynamic=true
export FLAGS_use_cinn=1

BS_LIST=${AP_BS_LIST:-"1 8 32"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TEST_FILES=(
    "test_ap_matmul_add_relu_iluvatar.py"
    "test_ap_matmul_add_multiply_iluvatar.py"
    "test_ap_matmul_add_gelu_iluvatar.py"
    "test_ap_matmul_add_divide_multiply_iluvatar.py"
    "test_ap_matmul_add_divide_multiply_add_iluvatar.py"
)

trap 'rm -f "${SCRIPT_DIR}"/.tmp_*_bs*.py' EXIT

for test_file in "${TEST_FILES[@]}"; do
    test_path="${SCRIPT_DIR}/${test_file}"

    for bs in ${BS_LIST}; do
        echo "========== Running ${test_file} with BS=${bs} =========="
        rm -rf /tmp/paddle/ap_workspace/*

        tmp_test="${SCRIPT_DIR}/.tmp_${test_file%.py}_bs${bs}.py"
        cp "${test_path}" "${tmp_test}"

        python - "${tmp_test}" "${bs}" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
bs = sys.argv[2]
text = path.read_text()
text, count = re.subn(
    r"^BS\s*=\s*.*$",
    f"BS = {bs}",
    text,
    count=1,
    flags=re.MULTILINE,
)
if count != 1:
    raise SystemExit(f"Failed to update BS in {path}")
path.write_text(text)
PY

        python "${tmp_test}"
        echo "========== Finished ${test_file} with BS=${bs} =========="
    done
done
