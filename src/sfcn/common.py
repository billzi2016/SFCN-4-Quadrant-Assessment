from __future__ import annotations

"""
SFCN 项目公共配置、路径、年龄分箱和 CSV I/O 工具。

本文件是全项目“口径中心”：
- 路径从 `config.yaml` 读取，避免脚本里到处硬编码。
- 年龄分箱、generated 文件名解析、DataLoader worker 策略都在这里统一。
- 训练、推理、画图和 manifest 生成都依赖这里的函数保持一致。

维护原则：任何会影响实验口径的常量和规则，优先放在这里集中修改。
"""

import csv
import json
import math
import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import yaml

# 这份文件集中放全项目共用的常量、年龄映射规则和简单 I/O 工具。
# 这样后续如果要改年龄范围、输出目录或 generated 年龄解析逻辑，
# 不需要去多个脚本里分别找。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_project_config(path: Path = CONFIG_PATH) -> dict:
    """读取项目 YAML 配置；缺失字段由代码默认值兜底。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    return loaded


PROJECT_CONFIG = load_project_config()


def config_section(name: str) -> dict:
    """安全读取 YAML 中的某个 section。"""
    section = PROJECT_CONFIG.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"配置 section 必须是 dict: {name}")
    return section


PATH_CONFIG = config_section("paths")
TRAINING_CONFIG = config_section("training")
DATA_LOADER_CONFIG = config_section("data_loader")

REAL_ROOT = Path(PATH_CONFIG.get("real_root", "/Volumes/LuZhang16T/IU_Datasets"))
REAL_MAPPING_TABLE = REAL_ROOT / "mapping_table.csv"
GENERATED_ROOT = Path(PATH_CONFIG.get("generated_root", "/Volumes/LuZhang16T/generated_mri"))
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR = OUTPUT_ROOT / "logs"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
FIGURE_DIR = OUTPUT_ROOT / "figures"
EXPERIMENT_CONFIGS = config_section("experiments")
QUADRANT_CONFIGS = config_section("quadrants")
EXPERIMENTS = tuple(EXPERIMENT_CONFIGS.keys()) or ("real-gen", "gen-real")

MIN_AGE = 5
MAX_AGE = 105
AGE_BIN_SIZE = 5
GENERATED_CONDITION_MIN = 0.0
GENERATED_CONDITION_MAX = 1.0
VALIDATION_SAMPLES_PER_AGE_SEX = 10

def resolve_num_workers(value: int | str | None = None) -> int:
    """解析 DataLoader worker 数；macOS 默认 0，Linux 默认 CPU 核数的 1/4。"""
    configured = DATA_LOADER_CONFIG.get("num_workers", "auto") if value is None else value
    if isinstance(configured, int):
        return max(configured, 0)
    if str(configured).lower() != "auto":
        return max(int(configured), 0)
    # macOS + MPS + nibabel + PyTorch DataLoader 多进程在本机出现过训练结束卡住。
    # 因此 Darwin 默认关闭 worker；Linux 没有这个已知问题，继续用多 worker 加速。
    if platform.system() == "Darwin":
        return 0
    cpu_count = os.cpu_count() or 1
    return max(cpu_count // 4, 1)


DEFAULT_BATCH_SIZE = int(TRAINING_CONFIG.get("batch_size", 8))
DEFAULT_NUM_WORKERS = resolve_num_workers()
DEFAULT_MAX_EPOCHS = int(TRAINING_CONFIG.get("max_epochs", 50))
DEFAULT_PATIENCE = int(TRAINING_CONFIG.get("patience", 2))
DEFAULT_LR = float(TRAINING_CONFIG.get("learning_rate", 3e-4))
DEFAULT_WEIGHT_DECAY = float(TRAINING_CONFIG.get("weight_decay", 1e-4))
DEFAULT_LOG_INTERVAL = int(TRAINING_CONFIG.get("log_interval", 50))
DEFAULT_DEVICE = str(TRAINING_CONFIG.get("device", "mps"))
DEFAULT_SEED = int(TRAINING_CONFIG.get("seed", 42))

EXPECTED_SHAPE = (182, 218, 182)

# generated MRI 目前已知有两种文件名风格：
# 1. age1.00_sexM_s131.nii.gz
# 2. 0004_age0.00_sexF_s4.nii.gz
# 前缀数字没有语义，因此这里显式写成可选项并直接忽略。
GENERATED_NAME_PATTERN = re.compile(
    r"^(?:\d+_)?age(?P<age>\d+(?:\.\d+)?)_sex(?P<sex>[MF])_s(?P<sample>\d+)\.nii(?:\.gz)?$"
)


@dataclass(frozen=True)
class TrainingConfig:
    # 这个配置类的目标不是做复杂配置系统，
    # 而是把“这次训练到底按什么规则跑”的核心参数持久化下来。
    # 后面保存到 json 里，方便回溯。
    real_root: str = str(REAL_ROOT)
    real_mapping_table: str = str(REAL_MAPPING_TABLE)
    generated_root: str = str(GENERATED_ROOT)
    output_root: str = str(OUTPUT_ROOT)
    min_age: int = MIN_AGE
    max_age: int = MAX_AGE
    age_bin_size: int = AGE_BIN_SIZE
    validation_samples_per_age_sex: int = VALIDATION_SAMPLES_PER_AGE_SEX
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    learning_rate: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    log_interval: int = DEFAULT_LOG_INTERVAL
    device: str = DEFAULT_DEVICE
    seed: int = DEFAULT_SEED
    expected_shape: tuple[int, int, int] = EXPECTED_SHAPE


def ensure_output_dirs() -> None:
    """创建旧版全局输出目录；新实验更多使用 ensure_experiment_dirs。"""
    # 统一在这里创建输出目录，避免每个脚本都各自 mkdir 一次。
    for path in (
        OUTPUT_ROOT,
        MANIFEST_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        PREDICTION_DIR,
        FIGURE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def experiment_root(experiment: str) -> Path:
    """根据实验名找到输出根目录，优先使用 config.yaml 中的 output_dir。"""
    config = get_experiment_config(experiment)
    output_dir = Path(config.get("output_dir", experiment))
    if output_dir.is_absolute():
        return output_dir
    return PROJECT_ROOT / output_dir


def get_experiment_config(experiment: str) -> dict:
    """读取某个实验的 YAML 配置。"""
    if experiment in EXPERIMENT_CONFIGS:
        config = EXPERIMENT_CONFIGS[experiment]
        if not isinstance(config, dict):
            raise ValueError(f"experiment 配置必须是 dict: {experiment}")
        return config
    if experiment in {"real-gen", "gen-real"}:
        # 兼容旧脚本和旧测试；新四象限实验应使用 config.yaml 中的 q* 名称。
        return {"output_dir": f"outputs/{experiment}"}
    raise ValueError(f"未知 experiment={experiment}, expected={tuple(EXPERIMENTS)}")


def experiment_dirs(experiment: str) -> dict[str, Path]:
    """返回某个实验固定使用的一组输出子目录。"""
    root = experiment_root(experiment)
    return {
        "root": root,
        "manifests": root / "manifests",
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "predictions": root / "predictions",
        "figures": root / "figures",
    }


def ensure_experiment_dirs(experiment: str | None = None) -> None:
    """创建一个或所有配置实验的 manifests/checkpoints/logs/predictions/figures 目录。"""
    experiments = EXPERIMENTS if experiment is None else (experiment,)
    for name in experiments:
        for path in experiment_dirs(name).values():
            path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    """写 JSON 文件，默认使用 UTF-8 和缩进，便于人工审阅。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_config(path: Path, config: TrainingConfig) -> None:
    """把训练配置 dataclass 保存成 JSON，便于复现实验。"""
    write_json(path, asdict(config))


def age_bin_edges(min_age: int = MIN_AGE, max_age: int = MAX_AGE, step: int = AGE_BIN_SIZE) -> list[int]:
    """返回年龄 bin 的边界，例如 5, 10, ..., 105。"""
    return list(range(min_age, max_age + 1, step))


def num_age_bins(min_age: int = MIN_AGE, max_age: int = MAX_AGE, step: int = AGE_BIN_SIZE) -> int:
    """返回年龄分类任务的类别数。"""
    return (max_age - min_age) // step


def age_to_bin_index(age_year: float, min_age: int = MIN_AGE, max_age: int = MAX_AGE, step: int = AGE_BIN_SIZE) -> int:
    # 年龄段采用 5 岁一档。
    # 这里的规则是：
    # - 常规区间左闭右开
    # - 最后一个区间把上界并进去
    clipped = min(max(age_year, float(min_age)), float(max_age))
    if math.isclose(clipped, float(max_age)):
        return num_age_bins(min_age, max_age, step) - 1
    return int((clipped - min_age) // step)


def bin_index_to_label(bin_index: int, min_age: int = MIN_AGE, step: int = AGE_BIN_SIZE) -> str:
    """把年龄 bin index 转成图表和 CSV 里使用的标签。"""
    lower = min_age + bin_index * step
    upper = lower + step
    return f"{lower}-{upper}"


def bin_index_to_center(bin_index: int, min_age: int = MIN_AGE, step: int = AGE_BIN_SIZE) -> float:
    """返回年龄 bin 中心点，用于把分类结果转换成年龄误差。"""
    lower = min_age + bin_index * step
    upper = lower + step
    return (lower + upper) / 2.0


def generated_age_to_year(
    age_value: float,
    min_age: int = MIN_AGE,
    max_age: int = MAX_AGE,
    cond_min: float = GENERATED_CONDITION_MIN,
    cond_max: float = GENERATED_CONDITION_MAX,
) -> int:
    """把 generated 文件名中的连续条件年龄映射回真实年龄年数。"""
    # generated 文件里的 age 目前不是“真实世界 header 年龄”，
    # 而是扩散模型条件值。这里按线性条件轴把它映射回 5~105 岁。
    # 当前假设是 0.00~1.00 对应 5~105 岁，共 101 个整数年龄点。
    ratio = (age_value - cond_min) / (cond_max - cond_min)
    ratio = min(max(ratio, 0.0), 1.0)
    return int(round(min_age + ratio * (max_age - min_age)))


def parse_generated_filename(path: Path) -> dict[str, str | float | int]:
    """解析 generated NIfTI 文件名，生成统一 manifest 行。"""
    # 这里是 generated 数据标签的唯一入口。
    # 后续任何脚本都不应该自己再重复写一套正则解析，
    # 否则年龄或 sample_id 的口径很容易漂。
    match = GENERATED_NAME_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"无法解析 generated 文件名: {path.name}")
    age_raw = float(match.group("age"))
    age_year = generated_age_to_year(age_raw)
    age_bin = age_to_bin_index(age_year)
    return {
        "path": str(path),
        "sex": match.group("sex"),
        "sample_id": int(match.group("sample")),
        "age_raw": age_raw,
        "age_year": age_year,
        "age_bin": age_bin,
        "age_bin_label": bin_index_to_label(age_bin),
        "source": "generated",
    }


def write_rows_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """写一组 dict 行到 CSV；fieldnames 缺省时按第一行字段顺序。"""
    # manifest、预测结果和统计结果都走同一套简单 CSV 输出，
    # 这样后面你直接用 pandas 或 Excel 看都方便。
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为 dict 列表；训练和推理 manifest 都走这个入口。"""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_by(rows: Iterable[dict], key: str) -> dict[str, int]:
    """按某个字段统计数量，返回字符串化 key 到 count 的映射。"""
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts
