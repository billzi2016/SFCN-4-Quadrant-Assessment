from __future__ import annotations

"""
SFCN 项目的唯一 Python 命令入口。

这个文件只负责做一件事：把用户输入的高层命令转发到具体模块。
这样 shell 脚本只需要调用 `python3 main.py <command>`，不用关心
真实实现文件在 `src/sfcn/` 下面的哪个模块里。

注意：这里不写训练、推理、画图逻辑，避免入口文件越来越胖。
"""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    """解析一级命令，并把剩余参数原样保留给下游模块。"""
    parser = argparse.ArgumentParser(description="SFCN four-quadrant experiment entrypoint")
    parser.add_argument(
        "command",
        choices=["build-manifests", "train", "infer", "plot", "scan-status"],
    )
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    """根据一级命令动态导入对应模块，并模拟原始 argv 后执行。"""
    parsed = parse_args()
    if parsed.command == "build-manifests":
        from sfcn import build_manifests as module
    elif parsed.command == "train":
        from sfcn import train as module
    elif parsed.command == "infer":
        from sfcn import infer as module
    elif parsed.command == "plot":
        from sfcn import evaluate_plot as module
    elif parsed.command == "scan-status":
        from sfcn import scan_status as module
    else:
        raise ValueError(f"unknown command: {parsed.command}")

    # 下游模块都有自己的 argparse。
    # 这里重写 sys.argv，是为了让 `train.py`、`infer.py` 等模块
    # 仍然像被直接运行一样解析参数，同时保留统一入口。
    sys.argv = [f"main.py {parsed.command}", *parsed.args]
    module.main()


if __name__ == "__main__":
    main()
