#!/bin/sh
# SFCN 四象限全流程命令清单。
#
# 保留这个脚本用于 Linux/终端复制粘贴命令。
# 可以直接执行，但如果担心长任务卡住，建议打开本文件逐段运行。

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

# Q1: Separated Training + Matched Age Distribution
./scripts/run_q1.sh

# Q2: Separated Training + Generated-Balanced Distribution
./scripts/run_q2.sh

# Q3: Mixed Training + Matched Age Distribution
./scripts/run_q3.sh

# Q4: Mixed Training + Peak-Valley Balanced Distribution
./scripts/run_q4.sh

# Plot all completed experiments
./scripts/run_plot.sh
