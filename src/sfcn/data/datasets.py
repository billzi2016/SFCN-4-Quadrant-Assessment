from __future__ import annotations

"""
PyTorch Dataset：把 manifest 行转换成 SFCN 可训练的 3D 张量样本。

重要约定：
- NIfTI 文件本身已经裁剪并 z-score 标准化，所以这里不再做归一化。
- generated 文件的 header 不可信，训练只使用数组值和 manifest 标签。
- 读取失败时必须带上 index 和 path，方便定位损坏文件或硬盘 I/O 问题。
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from sfcn.common import EXPECTED_SHAPE


class NiftiClassificationDataset(Dataset):
    # 这个 Dataset 只负责三件事：
    # 1. 从 manifest 里拿路径和标签
    # 2. 读取 NIfTI 体数据
    # 3. 返回训练/推理统一格式的数据字典
    #
    # 它刻意不做复杂预处理，因为当前数据已经完成裁剪和 z-score。
    def __init__(self, rows: list[dict], expected_shape: tuple[int, int, int] = EXPECTED_SHAPE) -> None:
        """保存 manifest 行和期望体数据 shape。"""
        self.rows = rows
        self.expected_shape = expected_shape

    def __len__(self) -> int:
        """返回 manifest 中样本数。"""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | float | int]:
        """读取单个 NIfTI 样本，并返回训练/推理共用的数据字典。"""
        row = self.rows[index]
        path = Path(row["path"])
        try:
            image = nib.load(str(path))

            # generated 的 header 不可信，但数组 shape 是可信的。
            # 因此这里只按数组空间读取，不基于 affine / spacing 做任何操作。
            volume = np.asarray(image.dataobj, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"NIfTI 读取失败: index={index} path={path}") from exc
        if tuple(volume.shape) != tuple(self.expected_shape):
            raise ValueError(f"{path} shape={volume.shape} 与预期 {self.expected_shape} 不一致")

        # SFCN 输入格式是 [C, D, H, W]。
        # 原始 NIfTI 是 [D, H, W]，所以在最前面补一个单通道维度。
        tensor = torch.from_numpy(volume).unsqueeze(0)
        return {
            "image": tensor,
            "target": torch.tensor(int(row["age_bin"]), dtype=torch.long),
            "age_year": torch.tensor(float(row["age_year"]), dtype=torch.float32),
            "sex": row["sex"],
            "path": str(path),
            "sample_id": int(row.get("sample_id", -1)),
            "age_raw": float(row.get("age_raw", row["age_year"])),
            "age_bin_label": row["age_bin_label"],
        }
