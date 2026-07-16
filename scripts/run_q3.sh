#!/bin/sh
# Q3: Mixed Training + Matched Age Distribution.
# 训练一个 mixed 模型：q3_mixed。
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

python3 main.py build-manifests --quadrant q3

python3 main.py train --experiment q3_mixed
python3 main.py infer --experiment q3_mixed --split real_test
python3 main.py infer --experiment q3_mixed --split generated_test
