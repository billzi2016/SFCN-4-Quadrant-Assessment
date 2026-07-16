#!/bin/sh
# Q1: Separated Training + Matched Age Distribution.
# 训练两个模型：q1_real_gen 和 q1_gen_real。
# 已存在 checkpoint/prediction 时会自动跳过；需要覆盖时手动给 Python 命令加 --force。

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

python3 main.py build-manifests --quadrant q1

python3 main.py train --experiment q1_real_gen
python3 main.py infer --experiment q1_real_gen --split real_val
python3 main.py infer --experiment q1_real_gen --split generated_test

python3 main.py train --experiment q1_gen_real
python3 main.py infer --experiment q1_gen_real --split generated_val
python3 main.py infer --experiment q1_gen_real --split real_test
