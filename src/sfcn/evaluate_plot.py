from __future__ import annotations

"""
汇总当前实验的 validation/test 逐样本预测结果，并绘制按性别拆分的年龄段 MAE 图。

本脚本只做评估和画图，不重新跑模型，也不修改 manifest。
输入是 infer.py 生成的逐样本 prediction CSV。
输出是论文可用的 PNG/PDF 图，以及按 age_bin 聚合的 metrics CSV。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sfcn.common import PROJECT_ROOT, ensure_experiment_dirs, experiment_dirs, get_experiment_config


def parse_args() -> argparse.Namespace:
    """解析实验名称，例如 real-gen 或 gen-real。"""
    parser = argparse.ArgumentParser(description="汇总预测结果并绘制年龄段 MAE 图")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--force", action="store_true", help="覆盖已有 figures")
    return parser.parse_args()


def prediction_paths(experiment: str) -> list[Path]:
    """返回某个实验需要比较的 prediction 文件路径。

    新实验统一读取 config.yaml 的 splits。
    对 separated 实验，通常是 validation + cross-domain test；
    对 mixed 实验，通常是 real_test + generated_test。
    """
    prediction_dir = experiment_dirs(experiment)["predictions"]
    config = get_experiment_config(experiment)
    splits = config.get("splits", {})
    if isinstance(splits, dict) and splits:
        return [prediction_dir / f"{split}_predictions.csv" for split in splits]
    if experiment == "real-gen":
        return [
            prediction_dir / "real_val_predictions.csv",
            prediction_dir / "generated_test_predictions.csv",
        ]
    if experiment == "gen-real":
        return [
            prediction_dir / "generated_val_predictions.csv",
            prediction_dir / "real_test_predictions.csv",
        ]
    raise ValueError(f"未知 experiment={experiment}")


def source_order(experiment: str) -> list[str]:
    """固定图中曲线顺序，避免不同运行中 legend 顺序漂移。"""
    config = get_experiment_config(experiment)
    splits = config.get("splits", {})
    if isinstance(splits, dict) and splits:
        return list(splits.keys())
    if experiment == "real-gen":
        return ["real_val", "generated_test"]
    if experiment == "gen-real":
        return ["generated_val", "real_test"]
    raise ValueError(f"未知 experiment={experiment}")


def display_source_label(source_label: str) -> str:
    """把内部 source_label 转成论文图例使用的 Title Case 英文。

    这里不要直接把 `real_val` 这种内部字段放进图例；
    论文图中必须使用可读的英文名。
    """
    labels = {
        "real_train": "Real Train",
        "real_val": "Real Validation",
        "generated_test": "Generated Test",
        "generated_train": "Generated Train",
        "generated_val": "Generated Validation",
        "generated_eval": "Generated Evaluation",
        "real_test": "Real Test",
        "real_eval": "Real Evaluation",
    }
    return labels.get(source_label, source_label.replace("_", " ").title())


def source_base(source_label: str) -> str:
    """把 train/val/test label 归并到 real 或 generated 来源。"""
    if source_label.startswith("real"):
        return "real"
    if source_label.startswith("generated"):
        return "generated"
    return "other"


def source_color(source_label: str) -> str:
    """固定颜色：real 一律绿色，generated 一律橙红色。"""
    if source_base(source_label) == "real":
        return "#1b9e77"
    if source_base(source_label) == "generated":
        return "#d95f02"
    return "#7570b3"


def source_count_color(source_label: str, split: str) -> str:
    """固定样本数柱颜色：train 深一些，validation/test 浅一些。"""
    base = source_base(source_label)
    if base == "real" and split == "train":
        return "#74c476"
    if base == "real":
        return "#c7e9c0"
    if base == "generated" and split == "train":
        return "#fdae6b"
    if base == "generated":
        return "#fdd0a2"
    return "#dadaeb"


def source_stack_order(source_label: str) -> int:
    """控制柱状图上下顺序：generated 在下，real 在上。

    Matplotlib 的堆叠柱是按绘制顺序从下往上叠加的；
    论文图要求绿色 real 压在橙色 generated 上方，所以 real 的排序值最大。
    """
    base = source_base(source_label)
    if base == "generated":
        return 0
    if base == "real":
        return 2
    return 1


def split_stack_order(split: str) -> int:
    """控制同一来源内部上下顺序：浅色 eval/val/test 在下，深色 train 在上。"""
    if split == "train":
        return 1
    return 0


def line_draw_order(source_label: str) -> tuple[int, int]:
    """控制 MAE 折线覆盖顺序：generated 先画，real 后画。"""
    return (source_stack_order(source_label), 0 if "generated" in source_label else 1)


def training_manifest_path(experiment: str) -> Path:
    """返回当前实验训练域的 train manifest，用于统计 train count。

    图里的柱状图展示样本数构成，所以除了 prediction CSV，
    还要读取训练 manifest 来画 train count。
    """
    config = get_experiment_config(experiment)
    train_manifest = config.get("train_manifest")
    if train_manifest:
        path = Path(str(train_manifest))
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path
    manifest_dir = experiment_dirs(experiment)["manifests"]
    if experiment == "real-gen":
        return manifest_dir / "real_train_manifest.csv"
    if experiment == "gen-real":
        return manifest_dir / "generated_train_manifest.csv"
    raise ValueError(f"未知 experiment={experiment}")


def load_train_count_rows(experiment: str) -> list[dict]:
    """从 train manifest 读取训练域样本数，用于堆叠 count 柱。

    mixed 实验中 train manifest 同时包含 real 和 generated，
    因此这里按 source、sex、age_bin_label 三个维度聚合。
    """
    path = training_manifest_path(experiment)
    if not path.exists():
        raise FileNotFoundError(f"缺少 train manifest: {path}")

    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source", "")
            if source not in {"real", "generated"}:
                source = "generated" if "generated" in str(row.get("path", "")) else "real"
            grouped[(source, row["sex"], row["age_bin_label"])] += 1

    return [
        {
            "sex": sex,
            "source_label": f"{source}_train",
            "age_bin_label": age_bin_label,
            "count": count,
            "mae": "",
            "mse": "",
        }
        for (source, sex, age_bin_label), count in sorted(
            grouped.items(),
            key=lambda item: (item[0][1], item[0][0], int(item[0][2].split("-")[0])),
        )
    ]


def load_prediction_rows(paths: list[Path]) -> list[dict[str, str]]:
    """读取逐样本预测 CSV。缺失文件直接报错，避免画出不完整结果。"""
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"缺少预测文件: {path}")
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict]:
    """按 sex、source_label、age_bin_label 聚合 count、MAE 和 MSE。

    论文主图使用 MAE，因为它比 MSE 更容易解释；
    MSE 仍保存在 CSV 中，方便和训练日志或旧结果对照。
    """
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["sex"], row["source_label"], row["age_bin_label"])
        grouped[key].append(row)

    summary: list[dict] = []
    for (sex, source_label, age_bin_label), group_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], int(item[0][2].split("-")[0])),
    ):
        absolute_errors = [float(row["absolute_error"]) for row in group_rows]
        squared_errors = [float(row["squared_error"]) for row in group_rows]
        summary.append(
            {
                "sex": sex,
                "source_label": source_label,
                "age_bin_label": age_bin_label,
                "count": len(group_rows),
                "mae": sum(absolute_errors) / len(absolute_errors),
                "mse": sum(squared_errors) / len(squared_errors),
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict]) -> None:
    """保存聚合指标，供论文画图或人工检查。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sex", "source_label", "age_bin_label", "count", "mae", "mse"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sex(rows: list[dict], train_count_rows: list[dict], sex: str, experiment: str, output_path: Path) -> None:
    """绘制单个性别的年龄段 MAE 曲线，并用背景柱显示样本量。

    图形语义：
    - 折线表示 real/generated 在各年龄段上的 MAE。
    - 柱状图表示样本数量，用于解释某些年龄段误差是否受样本量影响。
    - mixed 训练时，训练柱按 Real Train + Generated Train 堆叠。
    """
    sex_rows = [row for row in rows if row["sex"] == sex]
    if not sex_rows:
        raise ValueError(f"{experiment} sex={sex} 没有可绘制数据")
    sex_train_rows = [row for row in train_count_rows if row["sex"] == sex]

    order = source_order(experiment)

    # x 轴必须包含训练和评估中出现过的所有 age bin。
    # 否则某些 bin 没有 prediction 但有 train count 时，柱状图会丢失上下文。
    all_labels = sorted(
        {row["age_bin_label"] for row in sex_rows} | {row["age_bin_label"] for row in sex_train_rows},
        key=lambda label: int(label.split("-")[0]),
    )
    x_positions = list(range(len(all_labels)))
    by_source: dict[str, dict[str, dict]] = {}
    for row in sex_rows:
        by_source.setdefault(row["source_label"], {})[row["age_bin_label"]] = row
    train_by_source: dict[str, dict[str, dict]] = {}
    for row in sex_train_rows:
        train_by_source.setdefault(row["source_label"], {})[row["age_bin_label"]] = row

    fig, ax_mae = plt.subplots(figsize=(13, 4.8))
    ax_count = ax_mae.twinx()

    # 左侧柱为训练相关样本数。非混合时 validation 会堆到训练侧；
    # 混合时 train 本身会按 Real/Generated 堆叠。
    train_bases = {source_base(label) for label in train_by_source}
    bar_segments: dict[str, list[tuple[str, str, dict[str, dict]]]] = {"left": [], "right": []}

    for train_label, rows_by_label in train_by_source.items():
        bar_segments["left"].append((train_label, "train", rows_by_label))

    for split_label in order:
        rows_by_label = by_source.get(split_label, {})
        # 非混合实验：validation 与 train 同域，堆在左侧训练柱上。
        # mixed 实验：real/generated test 通常放右侧，避免和 mixed train 堆叠混淆。
        side = "left" if source_base(split_label) in train_bases and len(train_bases) == 1 else "right"
        bar_segments[side].append((split_label, "eval", rows_by_label))

    for side, segments in bar_segments.items():
        bottom = [0 for _ in x_positions]
        x_offset = -0.18 if side == "left" else 0.18
        # 统一堆叠语义：橙色 generated 在下、绿色 real 在上；
        # 同一来源内部，浅色 eval/validation/test 在下，深色 train 在上。
        for source_label, split, rows_by_label in sorted(
            segments,
            key=lambda item: (source_stack_order(item[0]), split_stack_order(item[1]), item[0]),
        ):
            counts = [int(rows_by_label[label]["count"]) if label in rows_by_label else 0 for label in all_labels]
            ax_count.bar(
                [x + x_offset for x in x_positions],
                counts,
                width=0.32,
                bottom=bottom,
                color=source_count_color(source_label, split),
                alpha=0.58 if split == "train" else 0.5,
                label=f"{display_source_label(source_label)} Count",
                zorder=0,
            )
            bottom = [current + count for current, count in zip(bottom, counts)]

    # MAE 曲线只画 validation 和 test；train 只用于解释样本量构成。
    for source_label in sorted(order, key=line_draw_order):
        label_rows = by_source.get(source_label, {})
        line_x: list[int] = []
        line_y: list[float] = []
        for idx, label in enumerate(all_labels):
            row = label_rows.get(label)
            if row:
                line_x.append(idx)
                line_y.append(float(row["mae"]))

        ax_mae.plot(
            line_x,
            line_y,
            marker="o",
            linewidth=2.2,
            color=source_color(source_label),
            label=f"{display_source_label(source_label)} MAE",
            zorder=4 + source_stack_order(source_label),
        )

    ax_mae.set_xticks(x_positions)
    ax_mae.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=13)
    ax_mae.set_xlabel("Age Bin", fontsize=16)
    ax_mae.set_ylabel("MAE", fontsize=16)
    ax_count.set_ylabel("Sample Count", fontsize=16)
    ax_mae.grid(axis="y", linestyle="--", alpha=0.25)

    handles_left, labels_left = ax_mae.get_legend_handles_labels()
    handles_right, labels_right = ax_count.get_legend_handles_labels()
    # 图例放在图外上方，避免遮挡论文图中的曲线和柱子。
    ax_mae.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        fontsize=12,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"figure saved: {output_path}")
    print(f"figure saved: {output_path.with_suffix('.pdf')}")


def figures_complete(figure_dir: Path) -> bool:
    """检查当前实验的核心图文件是否都已存在，用于默认跳过重复画图。"""
    required = [
        figure_dir / "age_bin_metrics.csv",
        figure_dir / "male_age_bin_mae.png",
        figure_dir / "male_age_bin_mae.pdf",
        figure_dir / "female_age_bin_mae.png",
        figure_dir / "female_age_bin_mae.pdf",
    ]
    return all(path.exists() for path in required)


def main() -> None:
    """执行当前实验的评估汇总和男女分图绘制。"""
    args = parse_args()
    ensure_experiment_dirs(args.experiment)
    dirs = experiment_dirs(args.experiment)
    # 画图默认也防覆盖。需要重画论文图时显式使用 --force。
    if figures_complete(dirs["figures"]) and not args.force:
        print(f"skip plotting: figures already exist: {dirs['figures']}")
        print("use --force to overwrite")
        return
    rows = load_prediction_rows(prediction_paths(args.experiment))
    summary_rows = summarize(rows)
    train_count_rows = load_train_count_rows(args.experiment)

    summary_path = dirs["figures"] / "age_bin_metrics.csv"
    write_summary(summary_path, summary_rows)
    print(f"summary saved: {summary_path}")

    plot_sex(summary_rows, train_count_rows, "M", args.experiment, dirs["figures"] / "male_age_bin_mae.png")
    plot_sex(summary_rows, train_count_rows, "F", args.experiment, dirs["figures"] / "female_age_bin_mae.png")


if __name__ == "__main__":
    main()
