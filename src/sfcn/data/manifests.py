from __future__ import annotations

"""
旧版 manifest 构建辅助函数。

当前四象限实验主要使用 `sfcn.build_manifests`。
这个文件保留是为了兼容旧代码和旧测试：
- 从 mapping_table.csv 解析 real NIfTI 路径。
- 从 generated 文件名解析 validation/test manifest。

新实验逻辑不要在这里继续扩展，避免和 `build_manifests.py` 出现两套口径。
"""

import csv
import os
from pathlib import Path

from sfcn.common import (
    GENERATED_ROOT,
    MANIFEST_DIR,
    MIN_AGE,
    MAX_AGE,
    REAL_MAPPING_TABLE,
    REAL_ROOT,
    VALIDATION_SAMPLES_PER_AGE_SEX,
    age_to_bin_index,
    bin_index_to_label,
    parse_generated_filename,
    write_rows_csv,
)


def _index_real_files(real_root: Path) -> dict[str, list[str]]:
    """扫描 real_root，建立 basename 到候选文件路径的索引。"""
    # mapping_table.csv 里的 dir 字段来自服务器环境，
    # 和本地挂载路径不一定一一对应。
    # 所以先建立一个 basename -> 路径列表的索引，后面做回填解析。
    index: dict[str, list[str]] = {}
    for dirpath, _, filenames in os.walk(real_root):
        if "MNI_Templates" in dirpath:
            continue
        for filename in filenames:
            if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
                continue
            index.setdefault(filename, []).append(str(Path(dirpath) / filename))
    return index


def _resolve_real_path(row_dir: str, dataset: str, file_index: dict[str, list[str]], real_root: Path) -> str | None:
    """把 mapping_table.csv 中的路径解析成本机真实可读路径。"""
    # 路径解析优先级：
    # 1. CSV 里原始路径本身可用
    # 2. 把 /IU_Datasets/ 后面的相对后缀挂到本地 real_root
    # 3. 用 basename 在本地索引中查找
    original = Path(row_dir)
    if original.exists():
        return str(original)

    marker = "/IU_Datasets/"
    normalized = row_dir.replace("\\", "/")
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = real_root / suffix
        if candidate.exists():
            return str(candidate)

    basename = original.name
    matches = file_index.get(basename, [])
    if not matches:
        return None

    dataset_matches = [path for path in matches if f"/{dataset}/" in path.replace("\\", "/")]
    selected = dataset_matches[0] if dataset_matches else matches[0]
    return selected


def build_real_train_manifest(
    mapping_table: Path = REAL_MAPPING_TABLE,
    real_root: Path = REAL_ROOT,
    output_path: Path = MANIFEST_DIR / "real_train_manifest.csv",
) -> list[dict]:
    """构建旧版 real train manifest，并写入 output_path。"""
    # real manifest 是训练主清单。
    # 它把 mapping_table.csv 中的标签与本地真实 NIfTI 文件路径绑定起来。
    file_index = _index_real_files(real_root)
    rows: list[dict] = []
    skipped = 0
    with mapping_table.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for source_row in reader:
            age = float(source_row["age"])
            if age < MIN_AGE or age > MAX_AGE:
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
                    "sex": source_row["sex"],
                    "age_year": age,
                    "age_raw": age,
                    "age_bin": age_bin,
                    "age_bin_label": bin_index_to_label(age_bin),
                    "source": "real",
                }
            )
    write_rows_csv(output_path, rows)
    print(f"real manifest: kept={len(rows)} skipped={skipped} path={output_path}")
    return rows


def _collect_generated_rows(generated_root: Path = GENERATED_ROOT) -> list[dict]:
    """扫描 generated 目录并解析所有合法文件名。"""
    # generated 清单只靠文件名解析，不信 header，也不依赖额外元数据文件。
    rows: list[dict] = []
    for dirpath, _, filenames in os.walk(generated_root):
        for filename in sorted(filenames):
            if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
                continue
            path = Path(dirpath) / filename
            try:
                row = parse_generated_filename(path)
            except ValueError:
                continue
            rows.append(row)
    rows.sort(key=lambda row: (int(row["age_year"]), str(row["sex"]), int(row["sample_id"])))
    return rows


def build_generated_validation_manifest(
    generated_root: Path = GENERATED_ROOT,
    samples_per_age_sex: int = VALIDATION_SAMPLES_PER_AGE_SEX,
    output_path: Path = MANIFEST_DIR / "generated_val_manifest.csv",
) -> list[dict]:
    """构建旧版 generated validation manifest，每个年龄×性别固定取样。"""
    # validation 规则已经固定：
    # - 每个年龄
    # - 每个性别
    # - 精确取 10 张
    #
    # 如果某个组不足 10 张，直接报错，不做“有多少用多少”的退化。
    all_rows = _collect_generated_rows(generated_root)
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in all_rows:
        key = (int(row["age_year"]), str(row["sex"]))
        grouped.setdefault(key, []).append(row)

    selected: list[dict] = []
    for key in sorted(grouped):
        group = grouped[key]
        if len(group) < samples_per_age_sex:
            raise ValueError(
                f"generated validation 样本不足: age={key[0]} sex={key[1]} "
                f"available={len(group)} required={samples_per_age_sex}"
            )
        chosen = group[:samples_per_age_sex]
        selected.extend(chosen)
        print(
            f"generated val group age={key[0]} sex={key[1]} "
            f"selected={len(chosen)} available={len(group)}"
        )

    write_rows_csv(output_path, selected)
    print(f"generated val manifest: rows={len(selected)} path={output_path}")
    return selected


def build_generated_test_manifest(
    generated_root: Path = GENERATED_ROOT,
    output_path: Path = MANIFEST_DIR / "generated_test_manifest.csv",
) -> list[dict]:
    """构建旧版 generated test manifest，直接使用全量 generated。"""
    # test manifest 是全量 generated 清单，不做抽样。
    rows = _collect_generated_rows(generated_root)
    write_rows_csv(output_path, rows)
    print(f"generated test manifest: rows={len(rows)} path={output_path}")
    return rows
