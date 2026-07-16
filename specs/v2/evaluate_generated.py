from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from common import FIGURE_DIR, PREDICTION_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 generated 与 real_train 推理结果为按性别/年龄段的 MAE")
    parser.add_argument(
        "--generated-predictions",
        type=Path,
        default=PREDICTION_DIR / "generated_predictions.csv",
    )
    parser.add_argument(
        "--real-train-predictions",
        type=Path,
        default=PREDICTION_DIR / "real_train_predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
    )
    return parser.parse_args()


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sex", "source_label", "age_bin_label", "count", "mae"])
        writer.writeheader()
        writer.writerows(rows)


def load_grouped_errors(prediction_path: Path) -> dict[tuple[str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with prediction_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["sex"], row["source_label"], row["age_bin_label"])
            grouped[key].append(float(row["squared_error"]) ** 0.5)
    return grouped


def main() -> None:
    # 这个脚本只做统计汇总：
    # 从逐样本预测结果中，按来源、性别和年龄段聚合成 MAE 表。
    # 它不重新跑模型，保证评估与画图可以反复迭代。
    args = parse_args()
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in (args.generated_predictions, args.real_train_predictions):
        current = load_grouped_errors(path)
        for key, values in current.items():
            grouped[key].extend(values)

    summary_by_sex: dict[str, list[dict]] = {"M": [], "F": []}
    for (sex, source_label, label), values in sorted(grouped.items()):
        summary_by_sex[sex].append(
            {
                "sex": sex,
                "source_label": source_label,
                "age_bin_label": label,
                "count": len(values),
                "mae": sum(values) / len(values),
            }
        )

    male_path = args.output_dir / "male_age_bin_mae.csv"
    female_path = args.output_dir / "female_age_bin_mae.csv"
    write_summary(male_path, summary_by_sex["M"])
    write_summary(female_path, summary_by_sex["F"])
    print(f"male summary saved: {male_path}")
    print(f"female summary saved: {female_path}")


if __name__ == "__main__":
    main()
