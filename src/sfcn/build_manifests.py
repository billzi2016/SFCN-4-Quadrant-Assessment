from __future__ import annotations

"""
按四象限实验设计构建 SFCN manifests。

这个文件是实验设计落地的核心：它决定每个模型到底看哪些样本。
因此任何关于“样本数是否公平”“是否 matched”“是否 balanced”的改动，
都应该优先检查本文件，而不是训练代码。

核心原则：
- Q1/Q3 使用 matched age distribution。
- Q2 使用 generated-balanced distribution。
- Q4 使用 mixed peak-valley balanced distribution。
- 所有象限中 real 和 generated 总量必须严格一致，避免训练/评估规模失衡。
- Q2/Q4 在 generated-balanced 约束下尽量复用 Q1 generated 样本。
- 默认不覆盖已有 manifest；需要重建时显式传 --force。
"""

import argparse
import csv
import os
import random
from pathlib import Path

from sfcn.common import (
    DEFAULT_SEED,
    GENERATED_ROOT,
    MAX_AGE,
    MIN_AGE,
    OUTPUT_ROOT,
    REAL_MAPPING_TABLE,
    REAL_ROOT,
    age_to_bin_index,
    bin_index_to_label,
    parse_generated_filename,
    write_rows_csv,
)
from sfcn.data.manifests import _index_real_files, _resolve_real_path


def parse_args() -> argparse.Namespace:
    """解析 manifest 构建参数。

    `--quadrant all --force` 是重建全部数据来源的主命令。
    `--force` 只覆盖 manifest/summary，不会删除 checkpoint 或 prediction。
    """
    parser = argparse.ArgumentParser(description="构建 SFCN 四象限 manifests")
    parser.add_argument("--quadrant", choices=["q1", "q2", "q3", "q4", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="覆盖已有 manifest")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def normalize_sex(value: str) -> str | None:
    """把 mapping table 里的性别字段统一成 M/F。"""
    normalized = value.strip().upper()
    if normalized in {"M", "MALE"}:
        return "M"
    if normalized in {"F", "FEMALE"}:
        return "F"
    return None


def group_key(row: dict) -> tuple[int, str]:
    """返回平衡抽样和切分使用的 age_bin × sex 分组键。"""
    return int(row["age_bin"]), str(row["sex"])


def stable_shuffle(rows: list[dict], seed: int, label: str) -> list[dict]:
    """用固定 seed 和标签打乱，保证不同组合可复现且互不影响。"""
    shuffled = list(rows)
    rng = random.Random(f"{seed}:{label}")
    rng.shuffle(shuffled)
    return shuffled


def collect_real_rows(
    mapping_table: Path = REAL_MAPPING_TABLE,
    real_root: Path = REAL_ROOT,
) -> list[dict]:
    """从 mapping_table.csv 收集可用 real MRI，并解析成统一 manifest 行。

    这里会过滤年龄范围外、性别无法解析、路径无法定位的样本。
    返回值是“原始可用 real 池”，后面 Q1/Q2/Q3/Q4 还会继续抽样。
    """
    file_index = _index_real_files(real_root)
    rows: list[dict] = []
    skipped = 0
    with mapping_table.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for source_row in reader:
            try:
                age = float(source_row["age"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if age < MIN_AGE or age > MAX_AGE:
                skipped += 1
                continue
            sex = normalize_sex(source_row.get("sex", ""))
            if sex is None:
                skipped += 1
                continue
            resolved = _resolve_real_path(source_row["dir"], source_row["dataset"], file_index, real_root)
            if resolved is None:
                skipped += 1
                continue
            age_bin = age_to_bin_index(age)
            rows.append(
                {
                    "path": resolved,
                    "dataset": source_row["dataset"],
                    "subject_id": source_row["id"],
                    "sex": sex,
                    "sample_id": -1,
                    "age_year": age,
                    "age_raw": age,
                    "age_bin": age_bin,
                    "age_bin_label": bin_index_to_label(age_bin),
                    "source": "real",
                }
            )
    print(f"real collected rows={len(rows)} skipped={skipped}")
    return rows


def collect_generated_rows(generated_root: Path = GENERATED_ROOT) -> list[dict]:
    """扫描 generated MRI 文件，并通过统一文件名解析器生成 manifest 行。

    generated 的年龄、性别和 sample_id 全部来自文件名。
    这里不读取 NIfTI header，因为 generated header 不作为可信标签来源。
    """
    rows: list[dict] = []
    skipped = 0
    for dirpath, _, filenames in os.walk(generated_root):
        for filename in sorted(filenames):
            if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
                continue
            path = Path(dirpath) / filename
            try:
                row = dict(parse_generated_filename(path))
            except ValueError:
                skipped += 1
                continue
            row["dataset"] = ""
            row["subject_id"] = ""
            rows.append(row)
    rows.sort(key=lambda row: (int(row["age_bin"]), str(row["sex"]), float(row["age_year"]), int(row["sample_id"])))
    print(f"generated collected rows={len(rows)} skipped={skipped}")
    return rows


def group_rows(rows: list[dict]) -> dict[tuple[int, str], list[dict]]:
    """把 rows 按 age_bin × sex 分组。"""
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)
    return grouped


def select_by_quota(
    rows: list[dict],
    quotas: dict[tuple[int, str], int],
    seed: int,
    source_label: str,
) -> list[dict]:
    """按给定 quota 从每个 age_bin × sex 组合抽样。

    quota 的 key 是 `(age_bin, sex)`，value 是该组合要抽多少个。
    这个函数只负责执行 quota，不负责判断 quota 是否学术合理。
    """
    grouped = group_rows(rows)
    selected: list[dict] = []
    for key, quota in sorted(quotas.items()):
        if quota <= 0:
            continue
        group = grouped.get(key, [])
        selected.extend(stable_shuffle(group, seed, f"{source_label}:{key[0]}:{key[1]}")[:quota])
    return sorted(selected, key=lambda row: (int(row["age_bin"]), str(row["sex"]), str(row["path"])))


def select_by_quota_with_preferred(
    rows: list[dict],
    quotas: dict[tuple[int, str], int],
    seed: int,
    source_label: str,
    preferred_rows: list[dict],
) -> list[dict]:
    """按 quota 抽样，并在每组内优先复用 preferred_rows。

    Q2 generated-balanced 需要尽量复用 Q1 generated，减少不同实验之间
    无意义的数据差异；但如果 Q1 在某个组合不够 balanced quota，
    就从完整 generated 池里继续补。
    """
    grouped = group_rows(rows)
    preferred_paths = {str(row["path"]) for row in preferred_rows}
    selected: list[dict] = []
    for key, quota in sorted(quotas.items()):
        if quota <= 0:
            continue
        group = grouped.get(key, [])
        preferred = [row for row in group if str(row["path"]) in preferred_paths]
        fallback = [row for row in group if str(row["path"]) not in preferred_paths]
        preferred = stable_shuffle(preferred, seed, f"{source_label}:preferred:{key[0]}:{key[1]}")
        fallback = stable_shuffle(fallback, seed, f"{source_label}:fallback:{key[0]}:{key[1]}")
        selected.extend((preferred + fallback)[:quota])
    return sorted(selected, key=lambda row: (int(row["age_bin"]), str(row["sex"]), str(row["path"])))


def matched_quotas(real_rows: list[dict], generated_rows: list[dict], min_per_group: int = 2) -> dict[tuple[int, str], int]:
    """Q1/Q3: real 和 generated 使用相同 age_bin × sex 联合直方图。

    对每个组合取 `min(real_count, generated_count)`，
    这样 matched 后两边在年龄和性别上完全同分布。
    """
    real_grouped = group_rows(real_rows)
    generated_grouped = group_rows(generated_rows)
    quotas: dict[tuple[int, str], int] = {}
    for key in sorted(set(real_grouped) | set(generated_grouped)):
        quota = min(len(real_grouped.get(key, [])), len(generated_grouped.get(key, [])))
        quotas[key] = quota if quota >= min_per_group else 0
    return quotas


def generated_balanced_quotas(
    generated_rows: list[dict],
    target_total: int,
    seed: int,
    min_per_group: int = 2,
) -> dict[tuple[int, str], int]:
    """为 generated-balanced 象限生成 quota，总量严格等于 real 总量。

    Q2 的核心不是“generated 越多越好”，而是“generated 在年龄/性别上均衡”。
    所以这里先把 target_total 平均分到 40 个组合，再用 seed=42 固定余数。
    """
    grouped = group_rows(generated_rows)
    candidate_keys = sorted(key for key, group in grouped.items() if len(group) >= min_per_group)
    if not candidate_keys or target_total <= 0:
        return {}

    # 先平均分配，再把余数按固定随机种子分配到不同 age_bin × sex 组合。
    base = target_total // len(candidate_keys)
    remainder = target_total % len(candidate_keys)
    shuffled_keys = stable_shuffle([{"key": key} for key in candidate_keys], seed, "balanced_remainder")
    remainder_keys = {row["key"] for row in shuffled_keys[:remainder]}
    quotas = {key: base + (1 if key in remainder_keys else 0) for key in candidate_keys}

    # 如果某些组合可用样本不足，把缺口继续分配给还有余量的组合，保持总量不变。
    changed = True
    while changed:
        changed = False
        deficit = 0
        for key in candidate_keys:
            capacity = len(grouped[key])
            if quotas[key] > capacity:
                deficit += quotas[key] - capacity
                quotas[key] = capacity
                changed = True
        if deficit <= 0:
            break
        spare_keys = [key for key in candidate_keys if quotas[key] < len(grouped[key])]
        if not spare_keys:
            raise ValueError(f"generated 可用样本不足，无法满足 target_total={target_total}")
        for key in stable_shuffle([{"key": key} for key in spare_keys], seed + deficit, "balanced_deficit"):
            if deficit <= 0:
                break
            group_key_value = key["key"]
            quotas[group_key_value] += 1
            deficit -= 1
    if sum(quotas.values()) != target_total:
        raise ValueError(f"generated-balanced quota 总量错误: {sum(quotas.values())} != {target_total}")
    return quotas


def balanced_quotas_for_keys(keys: list[tuple[int, str]], target_total: int, seed: int, label: str) -> dict[tuple[int, str], int]:
    """把 target_total 尽量平均分配到指定 age_bin × sex keys。

    Q4 train/val/test 都需要自己的目标总量；
    这个函数负责把各 split 的总量拆成每个组合的目标数量。
    """
    if not keys or target_total <= 0:
        return {}
    sorted_keys = sorted(keys)
    base = target_total // len(sorted_keys)
    remainder = target_total % len(sorted_keys)
    shuffled_keys = stable_shuffle([{"key": key} for key in sorted_keys], seed, label)
    remainder_keys = {row["key"] for row in shuffled_keys[:remainder]}
    return {key: base + (1 if key in remainder_keys else 0) for key in sorted_keys}


def real_original_quotas(real_rows: list[dict], min_per_group: int = 2) -> dict[tuple[int, str], int]:
    """Q2/Q4: real 保持自身可用分布，只过滤过小组合。"""
    grouped = group_rows(real_rows)
    return {key: len(group) for key, group in grouped.items() if len(group) >= min_per_group}


def q1_base_selection(real_rows: list[dict], generated_rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """生成所有象限复用的 Q1 基准 real/generated 样本集合。

    这个基准池是后续数量对齐的锚点。当前真实数据中，原始 real
    和 generated 并不是每个组合都一样多，所以先 matched 后得到
    `real == generated` 的共同可用池。
    """
    quotas = matched_quotas(real_rows, generated_rows)
    real_selected = select_by_quota(real_rows, quotas, seed, "q1_real")
    generated_selected = select_by_quota(generated_rows, quotas, seed, "q1_generated")
    if len(real_selected) != len(generated_selected):
        raise ValueError(f"Q1 real/generated 总量不一致: {len(real_selected)} != {len(generated_selected)}")
    return real_selected, generated_selected


def split_three_way(rows: list[dict], seed: int, source_label: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """按 age_bin × sex 分层切成 40% train、10% val、50% test。

    这是保留的通用工具；当前 Q3 为了严格对齐 separated 总量，
    使用 split_by_group_quotas 做更精确的切分。
    """
    grouped = group_rows(rows)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    summary: list[dict] = []
    for age_bin, sex in sorted(grouped):
        group = stable_shuffle(grouped[(age_bin, sex)], seed, f"split3:{source_label}:{age_bin}:{sex}")
        total = len(group)
        train_count = round(total * 0.4)
        val_count = round(total * 0.1)
        if train_count + val_count > total:
            val_count = max(total - train_count, 0)
        test_count = total - train_count - val_count
        train_rows.extend(group[:train_count])
        val_rows.extend(group[train_count : train_count + val_count])
        test_rows.extend(group[train_count + val_count :])
        summary.append(
            {
                "source": source_label,
                "age_bin": age_bin,
                "age_bin_label": bin_index_to_label(age_bin),
                "sex": sex,
                "total": total,
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "status": "split",
            }
        )
    return train_rows, val_rows, test_rows, summary


def distribute_total_by_group(rows: list[dict], target_total: int, seed: int, label: str) -> dict[tuple[int, str], int]:
    """按原始分布把 target_total 精确分配到各 age_bin × sex 组合。

    Q3 mixed matched 要保留 matched 分布，但 train/val/test 总量又要和
    separated 实验一致。这里用原分布比例分配 quota，再用固定 seed
    处理小数余量，保证总数精确等于 target_total。
    """
    grouped = group_rows(rows)
    if target_total <= 0:
        return {key: 0 for key in grouped}
    if target_total > len(rows):
        raise ValueError(f"target_total={target_total} 大于 rows={len(rows)}")
    raw: dict[tuple[int, str], float] = {
        key: target_total * len(group) / len(rows)
        for key, group in grouped.items()
    }
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = target_total - sum(quotas.values())
    fractional = sorted(
        [{"key": key, "fraction": raw[key] - quotas[key]} for key in grouped],
        key=lambda item: (-float(item["fraction"]), str(item["key"])),
    )
    for item in stable_shuffle(fractional, seed, f"quota:{label}"):
        if remaining <= 0:
            break
        key = item["key"]
        if quotas[key] < len(grouped[key]):
            quotas[key] += 1
            remaining -= 1
    if sum(quotas.values()) != target_total:
        raise ValueError(f"quota 分配失败: {sum(quotas.values())} != {target_total}")
    return quotas


def split_by_group_quotas(
    rows: list[dict],
    train_quotas: dict[tuple[int, str], int],
    val_quotas: dict[tuple[int, str], int],
    seed: int,
    source_label: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """按指定 train/val quota 精确切分，剩余样本作为 test。

    这个函数用于 Q3：Real 和 Generated 分别按同一套 matched 分布口径切分，
    但 train/val 的总量由外部精确控制，避免 mixed 数据集变成 2 倍。
    """
    grouped = group_rows(rows)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    summary: list[dict] = []
    for age_bin, sex in sorted(grouped):
        group = stable_shuffle(grouped[(age_bin, sex)], seed, f"split_quota:{source_label}:{age_bin}:{sex}")
        train_count = train_quotas.get((age_bin, sex), 0)
        val_count = val_quotas.get((age_bin, sex), 0)
        if train_count + val_count > len(group):
            raise ValueError(f"split quota 超出组容量: key={(age_bin, sex)}")
        train_rows.extend(group[:train_count])
        val_rows.extend(group[train_count : train_count + val_count])
        test_rows.extend(group[train_count + val_count :])
        summary.append(
            {
                "source": source_label,
                "age_bin": age_bin,
                "age_bin_label": bin_index_to_label(age_bin),
                "sex": sex,
                "total": len(group),
                "train": train_count,
                "val": val_count,
                "test": len(group) - train_count - val_count,
                "status": "split",
            }
        )
    return train_rows, val_rows, test_rows, summary


def allocate_capped(total: int, caps: dict[str, int], weights: dict[str, int], seed: int, label: str) -> dict[str, int]:
    """把 total 按权重分到不同 split，并且不超过每个 split 的 cap。

    Q4 削峰填谷里，某个 age/sex 组合可能 real 很多，也可能 real 很少。
    caps 是每个 split 最多允许放多少 real；如果 real 不够，后面由
    generated 自动补齐到目标 target。
    """
    if total <= 0:
        return {key: 0 for key in caps}
    remaining = min(total, sum(caps.values()))
    allocation = {key: 0 for key in caps}
    active = [key for key, cap in caps.items() if cap > 0]
    while remaining > 0 and active:
        weight_sum = sum(max(weights[key], 1) for key in active)
        progressed = False
        ordered = stable_shuffle([{"key": key} for key in active], seed + remaining, f"{label}:alloc")
        for item in ordered:
            key = item["key"]
            if remaining <= 0:
                break
            if allocation[key] >= caps[key]:
                continue
            share = max(1, round(remaining * max(weights[key], 1) / weight_sum))
            add = min(share, caps[key] - allocation[key], remaining)
            allocation[key] += add
            remaining -= add
            progressed = progressed or add > 0
        active = [key for key in active if allocation[key] < caps[key]]
        if not progressed:
            break
    return allocation


def select_peak_valley_splits(
    real_rows: list[dict],
    generated_rows: list[dict],
    target_totals: dict[str, int],
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Q4: 对每个 split 做削峰填谷，保持 mixed 总量与 separated 对齐。

    处理顺序：
    1. 为 train/val/test 分别生成 balanced 的每组目标数量。
    2. 每个组合先计算理想 real_part，即目标数量的一半。
    3. real 超过 real_part 时下采样，这是“削峰”。
    4. real 不足 real_part 时全部保留，缺口由 generated 补，这是“填谷”。
    5. 最终每个 split 的 mixed 总量精确等于 target_totals。
    """
    generated_grouped = group_rows(generated_rows)
    real_grouped = group_rows(real_rows)
    keys = sorted(generated_grouped)
    split_targets = {
        split: balanced_quotas_for_keys(keys, total, seed, f"q4_target:{split}")
        for split, total in target_totals.items()
    }

    outputs = {
        "train": {"real": [], "generated": []},
        "val": {"real": [], "generated": []},
        "test": {"real": [], "generated": []},
    }
    summary: list[dict] = []
    for key in keys:
        age_bin, sex = key
        target_by_split = {split: split_targets[split].get(key, 0) for split in ("train", "val", "test")}
        ideal_real = {split: target_by_split[split] // 2 for split in ("train", "val", "test")}
        real_group = stable_shuffle(real_grouped.get(key, []), seed, f"q4_real:{age_bin}:{sex}")
        generated_group = stable_shuffle(generated_grouped.get(key, []), seed, f"q4_generated:{age_bin}:{sex}")
        # real_quota 决定每个 split 从 real 里拿多少。
        # 它永远不会超过 ideal_real；超过的真实样本会被下采样掉。
        real_quota = allocate_capped(
            len(real_group),
            ideal_real,
            target_by_split,
            seed,
            f"q4_real_quota:{age_bin}:{sex}",
        )
        # generated_quota 是填谷结果：target 中 real 没占到的部分全部由 generated 补齐。
        generated_quota = {split: target_by_split[split] - real_quota[split] for split in ("train", "val", "test")}
        if sum(generated_quota.values()) > len(generated_group):
            raise ValueError(f"Q4 generated 不足: key={key} need={sum(generated_quota.values())} available={len(generated_group)}")

        real_offset = 0
        generated_offset = 0
        for split in ("train", "val", "test"):
            real_count = real_quota[split]
            generated_count = generated_quota[split]
            outputs[split]["real"].extend(real_group[real_offset : real_offset + real_count])
            outputs[split]["generated"].extend(generated_group[generated_offset : generated_offset + generated_count])
            real_offset += real_count
            generated_offset += generated_count
            summary.append(
                {
                    "source": "mixed",
                    "age_bin": age_bin,
                    "age_bin_label": bin_index_to_label(age_bin),
                    "sex": sex,
                    "split": split,
                    "target": target_by_split[split],
                    "real": real_count,
                    "generated": generated_count,
                    "status": "peak_valley",
                }
            )

    mixed_train = sorted(outputs["train"]["real"] + outputs["train"]["generated"], key=lambda row: (str(row["source"]), int(row["age_bin"]), str(row["sex"]), str(row["path"])))
    mixed_val = sorted(outputs["val"]["real"] + outputs["val"]["generated"], key=lambda row: (str(row["source"]), int(row["age_bin"]), str(row["sex"]), str(row["path"])))
    real_test = sorted(outputs["test"]["real"], key=lambda row: (int(row["age_bin"]), str(row["sex"]), str(row["path"])))
    generated_test = sorted(outputs["test"]["generated"], key=lambda row: (int(row["age_bin"]), str(row["sex"]), str(row["path"])))
    return mixed_train, mixed_val, real_test, generated_test, summary


def split_train_val(rows: list[dict], seed: int, source_label: str) -> tuple[list[dict], list[dict], list[dict]]:
    """对训练域按 age_bin × sex 分层做 4:1 train/validation 切分。

    Q1/Q2 separated 实验使用这个函数。每个组合内部单独切分，
    这样 train 和 val 的年龄/性别比例尽量接近。
    """
    grouped = group_rows(rows)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    summary: list[dict] = []
    for age_bin, sex in sorted(grouped):
        group = stable_shuffle(grouped[(age_bin, sex)], seed, f"split:{source_label}:{age_bin}:{sex}")
        total = len(group)
        if total < 2:
            train_count = total
            val_count = 0
            status = "validation_missing_insufficient_count"
        else:
            val_count = max(1, round(total * 0.2))
            train_count = total - val_count
            status = "split"
        train_rows.extend(group[:train_count])
        val_rows.extend(group[train_count : train_count + val_count])
        summary.append(
            {
                "source": source_label,
                "age_bin": age_bin,
                "age_bin_label": bin_index_to_label(age_bin),
                "sex": sex,
                "total": total,
                "train": train_count,
                "val": val_count,
                "status": status,
            }
        )
    return train_rows, val_rows, summary


def write_summary(path: Path, rows: list[dict], fieldnames: list[str], force: bool) -> None:
    """写出 summary CSV，默认不覆盖。"""
    if path.exists() and not force:
        print(f"skip existing summary: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, rows: list[dict], force: bool) -> None:
    """写出 manifest，默认不覆盖。"""
    if path.exists() and not force:
        print(f"skip existing manifest: {path}")
        return
    write_rows_csv(path, rows)


def count_summary(real_rows: list[dict], generated_rows: list[dict]) -> list[dict]:
    """记录 real/generated 每个 age_bin × sex 的 selected 数量。"""
    real_grouped = group_rows(real_rows)
    generated_grouped = group_rows(generated_rows)
    rows: list[dict] = []
    for key in sorted(set(real_grouped) | set(generated_grouped)):
        age_bin, sex = key
        rows.append(
            {
                "age_bin": age_bin,
                "age_bin_label": bin_index_to_label(age_bin),
                "sex": sex,
                "real_selected": len(real_grouped.get(key, [])),
                "generated_selected": len(generated_grouped.get(key, [])),
            }
        )
    return rows


def write_separated(
    root: Path,
    real_selected: list[dict],
    generated_selected: list[dict],
    seed: int,
    force: bool,
) -> None:
    """写出 separated training 的 real-gen 和 gen-real manifests。"""
    real_gen_dir = root / "real-gen" / "manifests"
    gen_real_dir = root / "gen-real" / "manifests"

    real_train, real_val, real_split = split_train_val(real_selected, seed, "real")
    gen_train, gen_val, gen_split = split_train_val(generated_selected, seed, "generated")

    write_manifest(real_gen_dir / "real_train_manifest.csv", real_train, force)
    write_manifest(real_gen_dir / "real_val_manifest.csv", real_val, force)
    write_manifest(real_gen_dir / "generated_test_manifest.csv", generated_selected, force)
    write_summary(real_gen_dir / "split_summary.csv", real_split, ["source", "age_bin", "age_bin_label", "sex", "total", "train", "val", "status"], force)

    write_manifest(gen_real_dir / "generated_train_manifest.csv", gen_train, force)
    write_manifest(gen_real_dir / "generated_val_manifest.csv", gen_val, force)
    write_manifest(gen_real_dir / "real_test_manifest.csv", real_selected, force)
    write_summary(gen_real_dir / "split_summary.csv", gen_split, ["source", "age_bin", "age_bin_label", "sex", "total", "train", "val", "status"], force)

    for manifest_dir in (real_gen_dir, gen_real_dir):
        write_summary(
            manifest_dir / "source_age_sex_count_summary.csv",
            count_summary(real_selected, generated_selected),
            ["age_bin", "age_bin_label", "sex", "real_selected", "generated_selected"],
            force,
        )


def write_mixed(
    root: Path,
    mixed_train: list[dict],
    mixed_val: list[dict],
    real_test: list[dict],
    generated_test: list[dict],
    split_summary: list[dict],
    count_rows: list[dict],
    summary_fieldnames: list[str],
    force: bool,
) -> None:
    """写出 mixed training 的 shared manifests。"""
    manifest_dir = root / "manifests"
    write_manifest(manifest_dir / "mixed_train_manifest.csv", mixed_train, force)
    write_manifest(manifest_dir / "mixed_val_manifest.csv", mixed_val, force)
    write_manifest(manifest_dir / "real_test_manifest.csv", real_test, force)
    write_manifest(manifest_dir / "generated_test_manifest.csv", generated_test, force)
    write_summary(
        manifest_dir / "split_summary.csv",
        split_summary,
        summary_fieldnames,
        force,
    )
    write_summary(
        manifest_dir / "source_age_sex_count_summary.csv",
        count_rows,
        ["age_bin", "age_bin_label", "sex", "real_selected", "generated_selected"],
        force,
    )


def build_q1(q1_real: list[dict], q1_generated: list[dict], seed: int, force: bool) -> None:
    """Q1: separated + matched age distribution。"""
    write_separated(OUTPUT_ROOT / "q1_separated_matched", q1_real, q1_generated, seed, force)


def build_q2(q1_real: list[dict], q1_generated: list[dict], generated_rows: list[dict], seed: int, force: bool) -> None:
    """Q2: separated + generated-balanced distribution。"""
    quotas = generated_balanced_quotas(generated_rows, target_total=len(q1_real), seed=seed)
    generated_selected = select_by_quota_with_preferred(generated_rows, quotas, seed, "q2_generated", q1_generated)
    write_separated(OUTPUT_ROOT / "q2_separated_gen_balanced", q1_real, generated_selected, seed, force)


def build_q3(q1_real: list[dict], q1_generated: list[dict], seed: int, force: bool) -> None:
    """Q3: mixed + matched age distribution。"""
    separated_train, separated_val, _ = split_train_val(q1_real, seed, "q3_target")
    real_train_total = len(separated_train) // 2
    gen_train_total = len(separated_train) - real_train_total
    real_val_total = len(separated_val) // 2
    gen_val_total = len(separated_val) - real_val_total
    real_train_quotas = distribute_total_by_group(q1_real, real_train_total, seed, "q3_real_train")
    real_val_quotas = distribute_total_by_group(q1_real, real_val_total, seed, "q3_real_val")
    gen_train_quotas = distribute_total_by_group(q1_generated, gen_train_total, seed, "q3_generated_train")
    gen_val_quotas = distribute_total_by_group(q1_generated, gen_val_total, seed, "q3_generated_val")
    real_train, real_val, real_test, real_summary = split_by_group_quotas(q1_real, real_train_quotas, real_val_quotas, seed, "real")
    gen_train, gen_val, gen_test, gen_summary = split_by_group_quotas(q1_generated, gen_train_quotas, gen_val_quotas, seed, "generated")
    mixed_train = sorted(real_train + gen_train, key=lambda row: (str(row["source"]), int(row["age_bin"]), str(row["sex"]), str(row["path"])))
    mixed_val = sorted(real_val + gen_val, key=lambda row: (str(row["source"]), int(row["age_bin"]), str(row["sex"]), str(row["path"])))
    write_mixed(
        OUTPUT_ROOT / "q3_mixed_matched",
        mixed_train,
        mixed_val,
        real_test,
        gen_test,
        real_summary + gen_summary,
        count_summary(q1_real, q1_generated),
        ["source", "age_bin", "age_bin_label", "sex", "total", "train", "val", "test", "status"],
        force,
    )


def build_q4(q1_real: list[dict], q1_generated: list[dict], generated_rows: list[dict], seed: int, force: bool) -> None:
    """Q4: mixed + peak-valley balanced distribution。"""
    separated_train, separated_val, _ = split_train_val(q1_real, seed, "q4_target")
    target_totals = {
        "train": len(separated_train),
        "val": len(separated_val),
        "test": len(q1_real),
    }
    mixed_train, mixed_val, real_test, generated_test, split_summary = select_peak_valley_splits(
        q1_real,
        generated_rows,
        target_totals,
        seed,
    )
    write_mixed(
        OUTPUT_ROOT / "q4_mixed_peak_valley",
        mixed_train,
        mixed_val,
        real_test,
        generated_test,
        split_summary,
        count_summary(real_test, generated_test),
        ["source", "age_bin", "age_bin_label", "sex", "split", "target", "real", "generated", "status"],
        force,
    )


def main() -> None:
    """构建四象限 manifests。"""
    args = parse_args()
    real_rows = collect_real_rows()
    generated_rows = collect_generated_rows()
    q1_real, q1_generated = q1_base_selection(real_rows, generated_rows, args.seed)
    print(f"q1 base selected real={len(q1_real)} generated={len(q1_generated)}")
    selected = ("q1", "q2", "q3", "q4") if args.quadrant == "all" else (args.quadrant,)
    for quadrant in selected:
        print(f"building {quadrant}")
        if quadrant == "q1":
            build_q1(q1_real, q1_generated, args.seed, args.force)
        elif quadrant == "q2":
            build_q2(q1_real, q1_generated, generated_rows, args.seed, args.force)
        elif quadrant == "q3":
            build_q3(q1_real, q1_generated, args.seed, args.force)
        elif quadrant == "q4":
            build_q4(q1_real, q1_generated, generated_rows, args.seed, args.force)


if __name__ == "__main__":
    main()
