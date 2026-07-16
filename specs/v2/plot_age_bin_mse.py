from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from common import FIGURE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制男女分开的年龄段 MAE 图")
    parser.add_argument("--male-csv", type=Path, default=FIGURE_DIR / "male_age_bin_mae.csv")
    parser.add_argument("--female-csv", type=Path, default=FIGURE_DIR / "female_age_bin_mae.csv")
    parser.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_one(rows: list[dict[str, str]], title: str, output_path: Path) -> None:
    # 画图脚本单独拆出来，目的是让你后面只改图形样式，
    # 不用碰训练和推理主逻辑。
    fig, ax_mse = plt.subplots(figsize=(13, 4.8))
    ax_count = ax_mse.twinx()

    # 同一张图里既要看误差，也要看每个年龄段实际有多少样本。
    # 因此左轴画 MAE，右轴画 count。
    # count 用半透明柱子放在背景层，避免压过误差曲线。
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["source_label"], []).append(row)

    source_order = ["generated", "real_train"]
    colors = {
        "generated": "#d95f02",
        "real_train": "#1b9e77",
    }
    count_colors = {
        "generated": "#fdd0a2",
        "real_train": "#c7e9c0",
    }

    # 不同来源可能覆盖的年龄段数量不同，例如 female 可能出现 19 vs 20。
    # 因此这里不能假设两边标签长度一致，而是要先做并集对齐。
    all_labels = sorted(
        {row["age_bin_label"] for row in rows},
        key=lambda label: int(label.split("-")[0]),
    )
    x_positions = list(range(len(all_labels)))

    for source_label in source_order:
        source_rows = grouped.get(source_label)
        if not source_rows:
            continue
        by_label = {row["age_bin_label"]: row for row in source_rows}
        values: list[float] = []
        counts: list[int] = []
        line_x: list[int] = []
        line_y: list[float] = []
        for idx, label in enumerate(all_labels):
            row = by_label.get(label)
            counts.append(int(row["count"]) if row is not None else 0)
            if row is not None:
                value = float(row["mae"])
                values.append(value)
                line_x.append(idx)
                line_y.append(value)

        # 两组样本量用左右轻微错开的背景柱子表示，
        # 这样可以直接看到同一年龄段 generated 和 real_train 的样本数差异。
        bar_offset = -0.18 if source_label == "generated" else 0.18
        ax_count.bar(
            [x + bar_offset for x in x_positions],
            counts,
            width=0.32,
            color=count_colors[source_label],
            alpha=0.45,
            label=f"{source_label} count",
            zorder=0,
        )
        ax_mse.plot(
            line_x,
            line_y,
            marker="o",
            linewidth=2.2,
            color=colors[source_label],
            label=f"{source_label} mae",
            zorder=3,
        )

    if not all_labels:
        raise ValueError(f"{output_path} 没有可绘制的数据")

    ax_mse.set_xticks(x_positions)
    ax_mse.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=13)
    ax_mse.set_xlabel("Age Bin", fontsize=16)
    ax_mse.set_ylabel("MAE", fontsize=16)
    ax_count.set_ylabel("Sample Count", fontsize=16)
    ax_mse.tick_params(axis="y", labelsize=13)
    ax_count.tick_params(axis="y", labelsize=13)
    ax_mse.grid(axis="y", linestyle="--", alpha=0.25)

    # 双轴图例需要手动合并，否则左右轴各自只显示一半。
    handles_left, labels_left = ax_mse.get_legend_handles_labels()
    handles_right, labels_right = ax_count.get_legend_handles_labels()
    ax_mse.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        ncol=2,
        fontsize=12,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"figure saved: {output_path}")
    print(f"figure saved: {output_path.with_suffix('.pdf')}")


def main() -> None:
    args = parse_args()
    male_rows = load_rows(args.male_csv)
    female_rows = load_rows(args.female_csv)
    plot_one(male_rows, "Male Age-Bin MAE", args.output_dir / "male_age_bin_mae.png")
    plot_one(female_rows, "Female Age-Bin MAE", args.output_dir / "female_age_bin_mae.png")


if __name__ == "__main__":
    main()
