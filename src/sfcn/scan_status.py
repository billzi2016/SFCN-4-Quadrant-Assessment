from __future__ import annotations

"""
扫描 SFCN 实验输出状态，避免重复运行已经完成的长任务。

这个脚本只读 config.yaml 和文件系统，不训练、不推理、不画图。
用于在长时间训练前确认哪些 manifest、checkpoint、prediction、figure 还缺失。
"""

import argparse
from pathlib import Path

from sfcn.common import EXPERIMENT_CONFIGS, OUTPUT_ROOT, PROJECT_ROOT, get_experiment_config, experiment_dirs


def parse_args() -> argparse.Namespace:
    """解析扫描参数。"""
    parser = argparse.ArgumentParser(description="扫描 SFCN 四象限实验输出状态")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--experiment", default="", help="只扫描某个配置实验名")
    return parser.parse_args()


def resolve_path(relative_or_abs: str) -> Path:
    """把 config.yaml 中的相对路径解析为 SFCN 目录下路径。"""
    path = Path(relative_or_abs)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def check_path(path: Path) -> str:
    """返回单个路径状态。"""
    return "OK" if path.exists() else "MISSING"


def print_item(label: str, path: Path, only_missing: bool) -> bool:
    """打印一个文件状态；返回是否存在。

    `--only-missing` 模式下，已经存在的文件不打印，
    这样终端里只剩真正需要补跑的步骤。
    """
    exists = path.exists()
    if exists and only_missing:
        return True
    print(f"  {check_path(path):7s} {label}: {path}")
    return exists


def scan_experiment(experiment: str, only_missing: bool) -> None:
    """扫描单个 experiment 的 manifest、checkpoint、prediction 和 figures。

    这里不根据文件内容判断是否“正确”，只判断是否存在。
    如果 manifest 逻辑变了，仍然需要你主动用 build-manifests --force 重建。
    """
    config = get_experiment_config(experiment)
    dirs = experiment_dirs(experiment)
    print(f"\n[{experiment}]")

    train_manifest = config.get("train_manifest")
    val_manifest = config.get("val_manifest")
    if train_manifest:
        print_item("train_manifest", resolve_path(str(train_manifest)), only_missing)
    if val_manifest:
        print_item("val_manifest", resolve_path(str(val_manifest)), only_missing)

    splits = config.get("splits", {})
    if isinstance(splits, dict):
        for split, manifest in splits.items():
            print_item(f"split_manifest:{split}", resolve_path(str(manifest)), only_missing)
            print_item(f"prediction:{split}", dirs["predictions"] / f"{split}_predictions.csv", only_missing)

    print_item("checkpoint", dirs["checkpoints"] / "best_model.pt", only_missing)
    for figure_name in (
        "age_bin_metrics.csv",
        "male_age_bin_mae.png",
        "male_age_bin_mae.pdf",
        "female_age_bin_mae.png",
        "female_age_bin_mae.pdf",
    ):
        print_item(f"figure:{figure_name}", dirs["figures"] / figure_name, only_missing)


def main() -> None:
    """打印配置中每个实验的 completed/missing 状态。"""
    args = parse_args()
    if args.experiment:
        scan_experiment(args.experiment, args.only_missing)
        return

    if not EXPERIMENT_CONFIGS:
        print(f"no configured experiments; output_root={OUTPUT_ROOT}")
        return
    for experiment in EXPERIMENT_CONFIGS:
        scan_experiment(experiment, args.only_missing)


if __name__ == "__main__":
    main()
