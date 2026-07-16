from __future__ import annotations

"""
SFCN 年龄段分类模型定义。

本实现保留 SFCN 的核心结构：3D 卷积逐步下采样、全局池化、线性分类头。
考虑到本项目主要在 Apple MPS 上训练，代码避免使用可能不稳定的 3D pooling，
而是通过 stride=2 的 Conv3d 完成下采样。
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    # 这里用的是简化后的 SFCN 风格卷积块：
    # Conv3d + BN + ReLU。
    # 如果需要下采样，则直接把这层卷积的 stride 设为 2。
    # 这样可以避开 MPS 对 3D pooling 算子支持不完整的问题。
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        """构建一个 3D 卷积块；pool=True 时用 stride=2 下采样。"""
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=2 if pool else 1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 Conv3d + BatchNorm3d + ReLU。"""
        return self.block(x)


class SFCNClassifier(nn.Module):
    # 这个模型不是复刻原论文的每一个细节，
    # 而是保留 SFCN 的核心思路：
    # - 3D 卷积逐层提特征
    # - 逐步下采样
    # - 最后做全局池化
    # - 接分类头预测年龄段
    def __init__(self, num_classes: int, dropout: float = 0.5) -> None:
        """创建年龄段分类模型，num_classes 对应年龄 bin 数。"""
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=True),
            ConvBlock(256, 256, pool=False),
            nn.Conv3d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # 3D CNN 主体使用 Kaiming/He 初始化，匹配 ReLU 激活。
        # BatchNorm 使用标准恒等初始化，避免一开始改变特征尺度。
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，输入 shape 为 [B, 1, 182, 218, 182]。"""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)
