from __future__ import annotations

"""
对当前实验的 validation 或 cross-domain test split 做推理。

本脚本按 --experiment 和 --split 自动选择 manifest、checkpoint 和输出路径。
它不计算汇总图，只保存逐样本 prediction CSV；画图和聚合由 evaluate_plot.py 完成。
"""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sfcn.common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_NUM_WORKERS,
    PROJECT_ROOT,
    bin_index_to_center,
    ensure_experiment_dirs,
    experiment_dirs,
    get_experiment_config,
    num_age_bins,
    read_rows_csv,
    resolve_num_workers,
)
from sfcn.data.datasets import NiftiClassificationDataset
from sfcn.models.sfcn import SFCNClassifier
from sfcn.train import collate_batch, data_loader_kwargs, resolve_device


def parse_args() -> argparse.Namespace:
    """解析实验名称、split 和推理运行参数。"""
    parser = argparse.ArgumentParser(description="对指定实验 split 做推理并保存逐样本结果")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--force", action="store_true", help="覆盖已有 prediction CSV")
    return parser.parse_args()


def resolve_manifest_and_output(experiment: str, split: str) -> tuple[Path, Path, str]:
    """把 experiment/split 映射到具体 manifest、prediction 输出文件和 source label。

    新实验都从 config.yaml 的 splits 字段读取路径。
    旧 real-gen/gen-real 分支仅用于兼容历史脚本和测试。
    """
    dirs = experiment_dirs(experiment)
    prediction_dir = dirs["predictions"]
    config = get_experiment_config(experiment)
    splits = config.get("splits")
    if isinstance(splits, dict) and split in splits:
        return PROJECT_ROOT / str(splits[split]), prediction_dir / f"{split}_predictions.csv", split

    manifest_dir = dirs["manifests"]
    if experiment == "real-gen" and split == "val":
        return (
            manifest_dir / "real_val_manifest.csv",
            prediction_dir / "real_val_predictions.csv",
            "real_val",
        )
    if experiment == "real-gen" and split == "test":
        return (
            manifest_dir / "generated_test_manifest.csv",
            prediction_dir / "generated_test_predictions.csv",
            "generated_test",
        )
    if experiment == "gen-real" and split == "val":
        return (
            manifest_dir / "generated_val_manifest.csv",
            prediction_dir / "generated_val_predictions.csv",
            "generated_val",
        )
    if experiment == "gen-real" and split == "test":
        return (
            manifest_dir / "real_test_manifest.csv",
            prediction_dir / "real_test_predictions.csv",
            "real_test",
        )
    raise ValueError(f"未知 experiment/split: {experiment}/{split}")


def main() -> None:
    """加载最佳 checkpoint，对指定 split 保存逐样本预测结果。"""
    args = parse_args()
    args.num_workers = resolve_num_workers(args.num_workers)
    ensure_experiment_dirs(args.experiment)
    dirs = experiment_dirs(args.experiment)
    manifest_path, output_path, source_label = resolve_manifest_and_output(args.experiment, args.split)
    checkpoint_path = dirs["checkpoints"] / "best_model.pt"

    # 推理也默认防覆盖。prediction CSV 一旦存在，说明这个 split 已经跑过；
    # 除非显式 --force，否则不重复耗时读取 NIfTI 和跑模型。
    if output_path.exists() and not args.force:
        print(f"skip inference: prediction already exists: {output_path}")
        print("use --force to overwrite")
        return
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少 manifest: {manifest_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"缺少 checkpoint: {checkpoint_path}")

    rows = read_rows_csv(manifest_path)
    dataset = NiftiClassificationDataset(rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        **data_loader_kwargs(args.num_workers),
    )

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SFCNClassifier(num_classes=num_age_bins())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # prediction CSV 是后续所有画图和统计的唯一输入。
    # 这里同时保存分类结果、预测年龄段中心、绝对误差和平方误差，
    # 这样 evaluate_plot.py 不需要重新加载模型。
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        seen = 0
        total = len(dataset)
        start_time = time.time()
        with torch.no_grad():
            for step, batch in enumerate(loader, start=1):
                images = batch["image"].to(device)
                # 模型输出是 age-bin logits；取 softmax 后最大类作为预测年龄段。
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                pred_probs, pred_classes = probs.max(dim=1)
                for idx in range(images.shape[0]):
                    pred_class = int(pred_classes[idx].item())
                    pred_age_center = bin_index_to_center(pred_class)
                    age_year = float(batch["age_year"][idx].item())
                    # 论文图一般用 MAE 更直观；MSE 保留给详细统计或调试。
                    absolute_error = abs(pred_age_center - age_year)
                    writer.writerow(
                        {
                            "experiment": args.experiment,
                            "source_label": source_label,
                            "path": batch["path"][idx],
                            "sex": batch["sex"][idx],
                            "sample_id": batch["sample_id"][idx],
                            "age_raw": batch["age_raw"][idx],
                            "age_year": age_year,
                            "age_bin": int(batch["target"][idx].item()),
                            "age_bin_label": batch["age_bin_label"][idx],
                            "target_class": int(batch["target"][idx].item()),
                            "pred_class": pred_class,
                            "pred_prob": float(pred_probs[idx].item()),
                            "pred_age_center": pred_age_center,
                            "absolute_error": absolute_error,
                            "squared_error": absolute_error**2,
                        }
                    )
                seen = min(seen + images.shape[0], total)
                elapsed = max(time.time() - start_time, 1e-6)
                step_per_sec = step / elapsed
                eta_sec = max(len(loader) - step, 0) / max(step_per_sec, 1e-6)
                if step == 1 or step % args.log_interval == 0 or seen == total:
                    print(
                        f"infer experiment={args.experiment} split={args.split} "
                        f"progress={seen}/{total} step={step}/{len(loader)} eta={eta_sec/60.0:.1f}m"
                    )
    print(f"predictions saved: {output_path}")


if __name__ == "__main__":
    main()
