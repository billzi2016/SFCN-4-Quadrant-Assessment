#!/bin/sh
# 统一画图脚本。
# 画图阶段不使用 DataLoader 多进程；如果训练和推理都已完成，可以一次性画全部配置实验。

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [ "$(uname -s)" = "Darwin" ]; then
    : "${OMP_NUM_THREADS:=4}"
    : "${MKL_NUM_THREADS:=4}"
    : "${VECLIB_MAXIMUM_THREADS:=4}"
    export OMP_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS
fi

python3 main.py plot --experiment q1_real_gen
python3 main.py plot --experiment q1_gen_real
python3 main.py plot --experiment q2_real_gen
python3 main.py plot --experiment q2_gen_real
python3 main.py plot --experiment q3_mixed
python3 main.py plot --experiment q4_mixed
