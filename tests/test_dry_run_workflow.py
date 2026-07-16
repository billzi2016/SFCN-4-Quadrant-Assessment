from __future__ import annotations

"""
SFCN dry run 测试。

这些测试不读取真实 MRI，也不训练模型。
目标是验证：
- manifest 写入流程可以在临时目录完成。
- Q2 generated-balanced 保持总量一致。
- Q3/Q4 mixed split 与 separated 的 train/val/test 规模对齐。
- 画图流程能从最小 prediction CSV 写出 summary 和图片。
"""

import csv
import sys
from pathlib import Path

import pytest

SFCN_ROOT = Path(__file__).resolve().parents[1]
SFCN_SRC = SFCN_ROOT / "src"
if str(SFCN_SRC) not in sys.path:
    sys.path.insert(0, str(SFCN_SRC))

from sfcn import build_manifests, common, evaluate_plot, infer, train


def make_row(source: str, age_bin: int, sex: str, index: int) -> dict:
    """构造不依赖真实 MRI 文件的 manifest 行。

    path 故意写成 /tmp 下的假路径；测试只验证 manifest 和 CSV 逻辑，
    不触发 NIfTI 实际读取。
    """
    age_year = 5 + age_bin * 5 + 1
    return {
        "path": f"/tmp/{source}_{age_bin}_{sex}_{index}.nii.gz",
        "dataset": "dryrun" if source == "real" else "",
        "subject_id": f"sub-{age_bin}-{sex}-{index}" if source == "real" else "",
        "sex": sex,
        "sample_id": index if source == "generated" else -1,
        "age_year": age_year,
        "age_raw": age_year,
        "age_bin": age_bin,
        "age_bin_label": common.bin_index_to_label(age_bin),
        "source": source,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取测试生成的 CSV 文件。"""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def patch_output_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """把所有实验输出目录重定向到 pytest 临时目录。

    这样测试不会污染真实 `SFCN/outputs`，也不会误改用户已经跑出的结果。
    """
    original_get_experiment_config = common.get_experiment_config

    def fake_get_experiment_config(experiment: str) -> dict:
        if experiment == "real-gen":
            return {"output_dir": str(tmp_path / "outputs" / "real-gen")}
        if experiment == "gen-real":
            return {"output_dir": str(tmp_path / "outputs" / "gen-real")}
        return original_get_experiment_config(experiment)

    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(common, "get_experiment_config", fake_get_experiment_config)
    monkeypatch.setattr(build_manifests, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(evaluate_plot, "experiment_dirs", common.experiment_dirs)
    monkeypatch.setattr(evaluate_plot, "ensure_experiment_dirs", common.ensure_experiment_dirs)
    monkeypatch.setattr(evaluate_plot, "get_experiment_config", fake_get_experiment_config)
    monkeypatch.setattr(infer, "experiment_dirs", common.experiment_dirs)
    monkeypatch.setattr(infer, "ensure_experiment_dirs", common.ensure_experiment_dirs)
    monkeypatch.setattr(train, "experiment_dirs", common.experiment_dirs)
    monkeypatch.setattr(train, "ensure_experiment_dirs", common.ensure_experiment_dirs)


def test_build_manifests_dry_run_writes_balanced_experiment_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry run: 用合成行验证 separated manifests 会写入临时目录。"""
    patch_output_root(monkeypatch, tmp_path)
    real_rows: list[dict] = []
    generated_rows: list[dict] = []
    for age_bin in (0, 1):
        for sex in ("M", "F"):
            for index in range(5):
                real_rows.append(make_row("real", age_bin, sex, index))
                generated_rows.append(make_row("generated", age_bin, sex, index))

    quotas = build_manifests.matched_quotas(real_rows, generated_rows)
    real_selected = build_manifests.select_by_quota(real_rows, quotas, seed=42, source_label="real")
    generated_selected = build_manifests.select_by_quota(generated_rows, quotas, seed=42, source_label="generated")
    assert len(real_selected) == len(generated_selected) == 20

    root = tmp_path / "outputs" / "q1_separated_matched"
    build_manifests.write_separated(root, real_selected, generated_selected, seed=42, force=False)

    real_gen_dir = root / "real-gen" / "manifests"
    gen_real_dir = root / "gen-real" / "manifests"

    assert (real_gen_dir / "real_train_manifest.csv").exists()
    assert (real_gen_dir / "real_val_manifest.csv").exists()
    assert (real_gen_dir / "generated_test_manifest.csv").exists()
    assert (gen_real_dir / "generated_train_manifest.csv").exists()
    assert (gen_real_dir / "generated_val_manifest.csv").exists()
    assert (gen_real_dir / "real_test_manifest.csv").exists()

    real_train = read_csv(real_gen_dir / "real_train_manifest.csv")
    real_val = read_csv(real_gen_dir / "real_val_manifest.csv")
    generated_train = read_csv(gen_real_dir / "generated_train_manifest.csv")
    generated_val = read_csv(gen_real_dir / "generated_val_manifest.csv")

    assert len(real_train) == 16
    assert len(real_val) == 4
    assert len(generated_train) == 16
    assert len(generated_val) == 4

    # 每个 age_bin × sex 组合有 5 个样本，按 4:1 后 val 每组应有 1 个。
    val_keys = {(row["age_bin"], row["sex"]) for row in real_val}
    assert val_keys == {("0", "M"), ("0", "F"), ("1", "M"), ("1", "F")}


def test_generated_balanced_keeps_same_total_and_reuses_q1_samples() -> None:
    """dry run: Q2 generated-balanced 总量必须等于 real，并尽量复用 Q1 generated。"""
    real_rows: list[dict] = []
    generated_rows: list[dict] = []
    for age_bin in (0, 1):
        for sex in ("M", "F"):
            for index in range(3 if age_bin == 0 else 5):
                real_rows.append(make_row("real", age_bin, sex, index))
            for index in range(12):
                generated_rows.append(make_row("generated", age_bin, sex, index))

    q1_real, q1_generated = build_manifests.q1_base_selection(real_rows, generated_rows, seed=42)
    quotas = build_manifests.generated_balanced_quotas(generated_rows, target_total=len(q1_real), seed=42)
    balanced_generated = build_manifests.select_by_quota_with_preferred(
        generated_rows,
        quotas,
        seed=42,
        source_label="q2_generated",
        preferred_rows=q1_generated,
    )

    assert len(q1_real) == len(q1_generated) == len(balanced_generated)
    assert sum(quotas.values()) == len(q1_real)
    assert max(quotas.values()) - min(quotas.values()) <= 1

    q1_paths = {row["path"] for row in q1_generated}
    balanced_paths = {row["path"] for row in balanced_generated}
    assert q1_paths & balanced_paths


def test_mixed_splits_keep_same_total_with_peak_valley() -> None:
    """dry run: Q3/Q4 mixed split 总量必须与 separated train/val/test 对齐。

    这个测试覆盖两个最容易写错的点：
    - Q3 mixed 不能因为 real+generated 合并而变成 2 倍训练集。
    - Q4 peak-valley 必须削峰填谷后仍满足 split 目标总量。
    """
    real_rows: list[dict] = []
    generated_rows: list[dict] = []
    for age_bin in (0, 1):
        for sex in ("M", "F"):
            real_count = 8 if age_bin == 0 else 2
            for index in range(real_count):
                real_rows.append(make_row("real", age_bin, sex, index))
            for index in range(20):
                generated_rows.append(make_row("generated", age_bin, sex, index))

    q1_real, q1_generated = build_manifests.q1_base_selection(real_rows, generated_rows, seed=42)
    separated_train, separated_val, _ = build_manifests.split_train_val(q1_real, seed=42, source_label="target")
    real_train_total = len(separated_train) // 2
    gen_train_total = len(separated_train) - real_train_total
    real_val_total = len(separated_val) // 2
    gen_val_total = len(separated_val) - real_val_total
    real_train_quotas = build_manifests.distribute_total_by_group(q1_real, real_train_total, seed=42, label="q3_real_train")
    real_val_quotas = build_manifests.distribute_total_by_group(q1_real, real_val_total, seed=42, label="q3_real_val")
    gen_train_quotas = build_manifests.distribute_total_by_group(q1_generated, gen_train_total, seed=42, label="q3_generated_train")
    gen_val_quotas = build_manifests.distribute_total_by_group(q1_generated, gen_val_total, seed=42, label="q3_generated_val")
    real_train, real_val, real_test, _ = build_manifests.split_by_group_quotas(q1_real, real_train_quotas, real_val_quotas, seed=42, source_label="real")
    gen_train, gen_val, gen_test, _ = build_manifests.split_by_group_quotas(q1_generated, gen_train_quotas, gen_val_quotas, seed=42, source_label="generated")

    assert len(real_train + gen_train) == len(separated_train)
    assert len(real_val + gen_val) == len(separated_val)
    assert len(real_test + gen_test) == len(q1_real)

    mixed_train, mixed_val, q4_real_test, q4_generated_test, summary = build_manifests.select_peak_valley_splits(
        q1_real,
        generated_rows,
        {"train": len(separated_train), "val": len(separated_val), "test": len(q1_real)},
        seed=42,
    )
    assert len(mixed_train) == len(separated_train)
    assert len(mixed_val) == len(separated_val)
    assert len(q4_real_test) + len(q4_generated_test) == len(q1_real)
    assert sum(int(row["target"]) for row in summary if row["split"] == "train") == len(separated_train)
    assert sum(int(row["target"]) for row in summary if row["split"] == "val") == len(separated_val)
    assert sum(int(row["target"]) for row in summary if row["split"] == "test") == len(q1_real)


def write_prediction_csv(path: Path, source_label: str) -> None:
    """写一个最小可画图的 prediction CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "source_label",
        "path",
        "sex",
        "sample_id",
        "age_raw",
        "age_year",
        "age_bin",
        "age_bin_label",
        "target_class",
        "pred_class",
        "pred_prob",
        "pred_age_center",
        "absolute_error",
        "squared_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sex in ("M", "F"):
            for age_bin in (0, 1):
                writer.writerow(
                    {
                        "experiment": "real-gen",
                        "source_label": source_label,
                        "path": f"/tmp/{source_label}_{sex}_{age_bin}.nii.gz",
                        "sex": sex,
                        "sample_id": -1,
                        "age_raw": 5 + age_bin * 5,
                        "age_year": 5 + age_bin * 5,
                        "age_bin": age_bin,
                        "age_bin_label": common.bin_index_to_label(age_bin),
                        "target_class": age_bin,
                        "pred_class": age_bin,
                        "pred_prob": 0.9,
                        "pred_age_center": common.bin_index_to_center(age_bin),
                        "absolute_error": 1.0,
                        "squared_error": 1.0,
                    }
                )


def test_evaluate_plot_dry_run_writes_summary_and_figures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry run: 验证评估画图流程能从预测 CSV 写出 summary、PNG 和 PDF。"""
    patch_output_root(monkeypatch, tmp_path)
    common.ensure_experiment_dirs("real-gen")
    dirs = common.experiment_dirs("real-gen")

    real_train_rows = [make_row("real", age_bin, sex, index) for age_bin in (0, 1) for sex in ("M", "F") for index in range(4)]
    common.write_rows_csv(dirs["manifests"] / "real_train_manifest.csv", real_train_rows)
    write_prediction_csv(dirs["predictions"] / "real_val_predictions.csv", "real_val")
    write_prediction_csv(dirs["predictions"] / "generated_test_predictions.csv", "generated_test")

    monkeypatch.setattr(sys, "argv", ["evaluate_plot.py", "--experiment", "real-gen"])
    evaluate_plot.main()

    assert (dirs["figures"] / "age_bin_metrics.csv").exists()
    assert (dirs["figures"] / "male_age_bin_mae.png").exists()
    assert (dirs["figures"] / "male_age_bin_mae.pdf").exists()
    assert (dirs["figures"] / "female_age_bin_mae.png").exists()
    assert (dirs["figures"] / "female_age_bin_mae.pdf").exists()

    summary_rows = read_csv(dirs["figures"] / "age_bin_metrics.csv")
    assert {row["source_label"] for row in summary_rows} == {"real_val", "generated_test"}
