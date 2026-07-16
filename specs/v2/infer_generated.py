from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_NUM_WORKERS,
    MANIFEST_DIR,
    PREDICTION_DIR,
    bin_index_to_center,
    ensure_output_dirs,
    num_age_bins,
    read_rows_csv,
)
from data.datasets import NiftiClassificationDataset
from models.sfcn import SFCNClassifier
from train import collate_batch, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对指定 manifest 做推理并保存逐样本结果")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_DIR / "generated_test_manifest.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DIR / "best_model.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PREDICTION_DIR / "generated_predictions.csv",
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default="",
        help="结果来源标签，例如 generated 或 real_train；为空时优先使用 manifest 内 source 字段",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    return parser.parse_args()


def main() -> None:
    # 这个脚本只负责一件事：
    # 用训练好的最佳模型，对指定 manifest 做推理并保存逐样本结果。
    # 它既可以跑 generated，也可以跑 real_train，只要 manifest 格式一致即可。
    args = parse_args()
    ensure_output_dirs()
    rows = read_rows_csv(args.manifest)
    dataset = NiftiClassificationDataset(rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_batch,
    )

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SFCNClassifier(num_classes=num_age_bins())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    fieldnames = [
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
        "squared_error",
        "source_label",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        seen = 0
        total = len(dataset)
        infer_start_time = time.time()
        with torch.no_grad():
            for step, batch in enumerate(loader, start=1):
                images = batch["image"].to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                pred_probs, pred_classes = probs.max(dim=1)
                for idx in range(images.shape[0]):
                    pred_class = int(pred_classes[idx].item())
                    # 这里把预测类别恢复为年龄段中心值，后面算 MSE 会直接用到。
                    pred_age_center = bin_index_to_center(pred_class)
                    age_year = float(batch["age_year"][idx].item())
                    row_source = rows[seen + idx].get("source", "unknown")
                    source_label = args.source_label or row_source
                    writer.writerow(
                        {
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
                            "squared_error": (pred_age_center - age_year) ** 2,
                            "source_label": source_label,
                        }
                    )
                seen = min(seen + images.shape[0], total)
                elapsed = max(time.time() - infer_start_time, 1e-6)
                step_per_sec = step / elapsed
                remaining_steps = max(len(loader) - step, 0)
                eta_sec = remaining_steps / max(step_per_sec, 1e-6)
                print(
                    f"infer progress={seen}/{total} step={step}/{len(loader)} "
                    f"eta={eta_sec/60.0:.1f}m"
                )
    print(f"predictions saved: {args.output}")


if __name__ == "__main__":
    main()
