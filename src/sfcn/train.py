from __future__ import annotations

"""
训练任意一个 config.yaml 中定义的 SFCN 实验。

本文件只关心“给定 train manifest 和 val manifest 后如何训练模型”。
它不负责决定 Q1/Q2/Q3/Q4 的抽样规则；抽样全部由 build_manifests.py 完成。

当前任务形式：
- 模型输出年龄段分类 logits。
- 训练 loss 是 CrossEntropyLoss。
- early stopping 使用 validation loss，也就是分类交叉熵。
- 日志中额外计算 MSE，是为了后续论文分析年龄误差。
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from sfcn.common import (
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PATIENCE,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    LOG_DIR,
    PROJECT_ROOT,
    TrainingConfig,
    bin_index_to_center,
    count_by,
    ensure_experiment_dirs,
    experiment_dirs,
    get_experiment_config,
    num_age_bins,
    read_rows_csv,
    resolve_num_workers,
    save_config,
)
from sfcn.data.datasets import NiftiClassificationDataset
from sfcn.models.sfcn import SFCNClassifier


def parse_args() -> argparse.Namespace:
    """解析训练参数；默认值来自 config.yaml 和 common.py。"""
    # 这里保留少量最关键的命令行参数，
    # 目的是保证你后续直接跑的时候不需要改太多代码。
    parser = argparse.ArgumentParser(description="训练 torch 版 SFCN 年龄段分类模型")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="覆盖已有 checkpoint/log 重新训练")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    """解析训练设备；优先使用请求设备，不可用时按 CUDA/MPS/CPU 降级。"""
    normalized = device_name.lower()
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    if normalized == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def collate_batch(batch: list[dict]) -> dict:
    """把 Dataset 返回的样本字典合成一个 batch 字典。"""
    # Dataset 返回的是字典；这里手工拼 batch，
    # 这样既能堆叠 tensor，也能保留 path / sex 这些元信息。
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    age_year = torch.stack([item["age_year"] for item in batch], dim=0)
    return {
        "image": images,
        "target": targets,
        "age_year": age_year,
        "sex": [item["sex"] for item in batch],
        "path": [item["path"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "age_raw": [item["age_raw"] for item in batch],
        "age_bin_label": [item["age_bin_label"] for item in batch],
    }


def build_sampler(rows: list[dict]) -> WeightedRandomSampler:
    """构建按年龄段反频率加权的训练 sampler。

    注意这里平衡的是训练时被抽到的概率，不改变 manifest 本身的样本组成。
    这样可以减少高频年龄段对梯度方向的支配，同时保留实验设计中的真实数量。
    """
    # 训练时做年龄段均衡采样，避免高频年龄段把模型训练方向带偏。
    counts = count_by(rows, "age_bin")
    weights = [1.0 / counts[str(row["age_bin"])] for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_eval_metrics(logits: torch.Tensor, targets: torch.Tensor, age_year: torch.Tensor) -> dict[str, float]:
    """计算验证阶段的分类准确率和年龄 MSE。"""
    # 训练任务是分类，但最终分析仍然关心年龄误差。
    # 所以这里同时计算：
    # - 分类准确率
    # - 把预测类别映射成年龄段中心值后的 MSE
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    accuracy = float((preds == targets).float().mean().item())
    pred_age_center = torch.tensor(
        [bin_index_to_center(int(idx)) for idx in preds.detach().cpu().tolist()],
        device=age_year.device,
        dtype=torch.float32,
    )
    mse = float(torch.mean((pred_age_center - age_year) ** 2).item())
    return {"accuracy": accuracy, "mse": mse}


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    """在 validation loader 上评估 loss、accuracy 和按性别/年龄段的 MSE。"""
    # evaluate 负责产出完整验证指标。
    # early stopping 当前看 val_loss，
    # accuracy 和 MSE 仍然会打印并保存，方便后续分析。
    model.eval()
    losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_age_year: list[torch.Tensor] = []
    sex_sse: dict[str, list[float]] = {"M": [], "F": []}
    bin_sse: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            age_year = batch["age_year"].to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            losses.append(float(loss.item()))
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_age_year.append(age_year.cpu())

            preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().tolist()
            years = age_year.cpu().tolist()
            for pred, year, sex, label in zip(preds, years, batch["sex"], batch["age_bin_label"]):
                # 这里按“预测年龄段中心值 vs 实际年龄”的平方误差记账，
                # 后面可以直接按性别和年龄段聚合成 MSE。
                squared_error = (bin_index_to_center(int(pred)) - float(year)) ** 2
                sex_sse[sex].append(squared_error)
                bin_sse.setdefault(label, []).append(squared_error)

    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    age_year_tensor = torch.cat(all_age_year, dim=0)
    metrics = compute_eval_metrics(logits_tensor, targets_tensor, age_year_tensor)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["mse_male"] = float(np.mean(sex_sse["M"])) if sex_sse["M"] else 0.0
    metrics["mse_female"] = float(np.mean(sex_sse["F"])) if sex_sse["F"] else 0.0
    for label, values in sorted(bin_sse.items()):
        metrics[f"mse_bin_{label}"] = float(np.mean(values))
    return metrics


def append_log_row(path: Path, row: dict) -> None:
    """向训练日志 CSV 追加一行；首次写入时自动创建表头。"""
    # 日志用 CSV 持久化，便于后面直接读表分析。
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def manifest_paths_for_experiment(experiment: str) -> tuple[Path, Path]:
    """返回当前实验对应的 train/validation manifest 路径。"""
    config = get_experiment_config(experiment)
    train_manifest = config.get("train_manifest")
    val_manifest = config.get("val_manifest")
    if train_manifest and val_manifest:
        return PROJECT_ROOT / str(train_manifest), PROJECT_ROOT / str(val_manifest)

    manifest_dir = experiment_dirs(experiment)["manifests"]
    if experiment == "real-gen":
        return manifest_dir / "real_train_manifest.csv", manifest_dir / "real_val_manifest.csv"
    if experiment == "gen-real":
        return manifest_dir / "generated_train_manifest.csv", manifest_dir / "generated_val_manifest.csv"
    raise ValueError(f"experiment 缺少 train_manifest/val_manifest 配置: {experiment}")


def data_loader_kwargs(num_workers: int) -> dict:
    """集中管理 DataLoader 的保守参数，避免 macOS/MPS 多进程卡住。

    `prefetch_factor` 只有 num_workers > 0 时才能传给 DataLoader；
    macOS 默认 num_workers=0，所以这里必须条件化设置。
    """
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 1
    return kwargs


def main() -> None:
    """执行完整训练流程：读 manifest、建模型、训练、验证、早停、保存 checkpoint。"""
    args = parse_args()
    args.num_workers = resolve_num_workers(args.num_workers)
    set_seed(args.seed)
    ensure_experiment_dirs(args.experiment)
    dirs = experiment_dirs(args.experiment)
    checkpoint_path = dirs["checkpoints"] / "best_model.pt"
    # 防覆盖逻辑很重要：这些实验可能跑十几个小时。
    # 如果 checkpoint 已经存在，默认直接跳过，避免误触脚本造成重训。
    if checkpoint_path.exists() and not args.force:
        print(f"skip training: checkpoint already exists: {checkpoint_path}")
        print("use --force to retrain and overwrite")
        return

    config = TrainingConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed,
    )
    save_config(dirs["logs"] / "training_config.json", config)

    train_manifest_path, val_manifest_path = manifest_paths_for_experiment(args.experiment)
    if not train_manifest_path.exists() or not val_manifest_path.exists():
        raise FileNotFoundError(
            f"缺少 manifest，请先运行 python3 main.py build-manifests: "
            f"{train_manifest_path} / {val_manifest_path}"
        )

    train_rows = read_rows_csv(train_manifest_path)
    val_rows = read_rows_csv(val_manifest_path)
    print(f"experiment={args.experiment} train rows={len(train_rows)} val rows={len(val_rows)}")

    train_dataset = NiftiClassificationDataset(train_rows)
    val_dataset = NiftiClassificationDataset(val_rows)
    # 训练 loader 使用 WeightedRandomSampler，所以这里不能再设置 shuffle=True。
    # sampler 已经决定了每个 epoch 的样本顺序和抽样概率。
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=build_sampler(train_rows),
        collate_fn=collate_batch,
        **data_loader_kwargs(args.num_workers),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        **data_loader_kwargs(args.num_workers),
    )

    device = resolve_device(args.device)
    print(f"device={device} batch_size={args.batch_size} num_workers={args.num_workers}")
    model = SFCNClassifier(num_classes=num_age_bins()).to(device)
    # 训练目标是年龄段分类，因此 loss 是交叉熵。
    # 论文里更常报告 MAE/MSE，但它们是从分类结果映射回年龄中心后计算的指标。
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    log_path = dirs["logs"] / "train_log.csv"
    training_start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running_losses: list[float] = []
        seen = 0
        total = len(train_dataset)
        epoch_start_time = time.time()
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_losses.append(float(loss.item()))
            seen = min(seen + images.shape[0], total)
            # 按你的要求，这里不用 tqdm。
            # 只打印简单文本进度和当前 loss，避免长终端日志太难看。
            if step == 1 or step % args.log_interval == 0 or seen == total:
                elapsed_epoch = max(time.time() - epoch_start_time, 1e-6)
                step_per_sec = step / elapsed_epoch
                remaining_steps_epoch = max(len(train_loader) - step, 0)
                eta_epoch_sec = remaining_steps_epoch / max(step_per_sec, 1e-6)

                elapsed_total = max(time.time() - training_start_time, 1e-6)
                finished_epochs = (epoch - 1) + step / max(len(train_loader), 1)
                epoch_per_sec = finished_epochs / elapsed_total
                remaining_epochs = max(args.max_epochs - finished_epochs, 0.0)
                eta_total_sec = remaining_epochs / max(epoch_per_sec, 1e-6)
                print(
                    f"train epoch={epoch}/{args.max_epochs} "
                    f"progress={seen}/{total} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} "
                    f"eta_epoch={eta_epoch_sec/60.0:.1f}m "
                    f"eta_total={eta_total_sec/3600.0:.1f}h"
                )

        train_loss = float(np.mean(running_losses)) if running_losses else 0.0
        val_metrics = evaluate(model, val_loader, criterion, device)
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_mse": val_metrics["mse"],
            "val_mse_male": val_metrics["mse_male"],
            "val_mse_female": val_metrics["mse_female"],
        }
        append_log_row(log_path, log_row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_mse={val_metrics['mse']:.4f} "
            f"val_mse_male={val_metrics['mse_male']:.4f} "
            f"val_mse_female={val_metrics['mse_female']:.4f}"
        )
        for key in sorted(val_metrics):
            if key.startswith("mse_bin_"):
                print(f"{key}={val_metrics[key]:.4f}")

        # early stopping 的唯一依据是 val_loss。
        # 这样和训练目标一致，也避免用 MSE/MAE 作为另一个隐式优化目标。
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            # checkpoint 只保存当前最优模型，口径由 val_loss 决定。
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config.__dict__,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"best checkpoint saved: {checkpoint_path} val_loss={best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"early_stopping patience={patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("early stopping triggered")
                break


if __name__ == "__main__":
    main()
