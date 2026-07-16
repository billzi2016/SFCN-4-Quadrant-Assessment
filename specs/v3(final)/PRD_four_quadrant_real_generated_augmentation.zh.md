# PRD: Four-Quadrant Real/Generated SFCN Experiments

## 1. 背景

当前 `SFCN` 实验需要从单一的 `real-gen / gen-real` 对照，扩展为四种实验模式。核心问题不只是“generated MRI 能不能泛化”，还包括：

- generated 是否只作为独立训练域使用
- generated 是否作为 real 数据的 augmentation 使用
- generated 年龄分布应当模拟 real 分布，还是作为年龄均衡补充
- 不同分布策略对真实生产场景和实验室理想场景的影响

之前已有结果对应第一类实验：`real` 和 `generated` 在年龄分布上被做成同分布，且数量一致。这一类可以作为“同分布控制实验”，但不能代表 generated 年龄均衡 augmentation 的真实目的。

## 2. 两个实验维度

四种实验由两个二元维度组成。

### 2.1 是否混合训练

- `Separated Training`: real 和 generated 分别作为训练来源，不混在同一个 train set 中。
- `Mixed Training`: real 和 generated 合并为一个 train set，用于模拟 generated 作为 augmentation 加入训练数据的生产场景。

### 2.2 年龄分布策略

- `Matched Age Distribution`: real 和 generated 使用相同年龄/性别分布，直方图尽量一致，总量也一致。
- `Generated-Balanced Distribution`: generated 在所有 `age_bin × sex` 组合上数量相同或尽量相同，用于 separated 条件下测试均衡 generated 的影响。
- `Peak-Valley Balanced Distribution`: mixed 条件下对 real 执行削峰填谷；real 过多的组合下采样，real 不足的组合用 generated 补齐。

## 3. 四个象限

| 象限 | 是否混合训练 | 年龄分布策略 | 实验含义 |
| --- | --- | --- | --- |
| Q1 | No | Matched Age Distribution | 实验室控制条件：real 和 generated 同分布、同数量，分别训练和测试跨域泛化 |
| Q2 | No | Generated-Balanced Distribution | 分离训练条件：generated 年龄均衡，检验 generated 作为独立训练域/测试域时的年龄覆盖影响 |
| Q3 | Yes | Matched Age Distribution | 理想 augmentation：real 和 generated 同分布混合训练，再分别测试 real/gen |
| Q4 | Yes | Peak-Valley Balanced Distribution | 生产 augmentation：削峰填谷后混合训练，测试 real/gen |

## 4. 总体原则

- real 和 generated 不再默认互相决定年龄分布。
- 每个象限必须明确自己的抽样策略。
- 所有随机抽样必须使用 `seed = 42`。
- 所有 split 必须按 `age_bin × sex` 分层处理。
- 所有 manifest 必须写 summary，记录每个 `age_bin × sex × source` 的样本数。
- 不允许复制样本。
- 不允许使用 mock 样本补齐。
- 不允许在论文中声称“年龄均衡”，但实际代码使用了 real 分布截断 generated。

## 5. Q1: Separated + Matched Age Distribution

### 5.1 数据构建

Q1 是当前已有图的实验类型。

要求：

- real selected 和 generated selected 总量一致。
- real 和 generated 的 `age_bin × sex` 直方图一致或尽量一致。
- 对每个 `age_bin × sex` 组合，使用共同 quota。
- 共同 quota 可以由 real/generated 中较小可用数量决定。

### 5.2 训练与测试

需要两个方向：

- `real -> generated`
  - train/early-stop split: real selected 内部按 `age_bin × sex` 切分。
  - test split: generated selected。
- `generated -> real`
  - train/early-stop split: generated selected 内部按 `age_bin × sex` 切分。
  - test split: real selected。

### 5.3 输出目录建议

```text
outputs/q1_separated_matched/
├── real-gen/
└── gen-real/
```

## 6. Q2: Separated + Generated-Balanced Distribution

### 6.1 数据构建

Q2 不允许让 generated 按 real 年龄分布截断。

要求：

- generated selected 必须在每个 `age_bin × sex` 上数量相同或尽量相同。
- real selected 保持 real 自身分布，可以是全量可用 real，也可以是按预先定义规则抽样后的 real。
- 如果需要 real 和 generated 总量一致，只能通过全局总量控制实现，不能让 real 的逐年龄段计数决定 generated 的逐年龄段计数。

### 6.2 训练与测试

需要两个方向：

- `real -> generated-balanced`
  - train/early-stop split: real 内部按 `age_bin × sex` 切分。
  - test split: generated balanced selected。
- `generated-balanced -> real`
  - train/early-stop split: generated balanced selected 内部按 `age_bin × sex` 切分。
  - test split: real selected。

### 6.3 输出目录建议

```text
outputs/q2_separated_gen_balanced/
├── real-gen/
└── gen-real/
```

## 7. Q3: Mixed + Matched Age Distribution

### 7.1 数据构建

Q3 模拟理想 augmentation 条件。

要求：

- real selected 和 generated selected 使用 matched age distribution。
- 两者混合为一个 train pool，但 mixed train 总量必须与 separated train 对齐，不允许简单翻倍。
- mixed train pool 中每个 `age_bin × sex × source` 的数量应写入 summary。

### 7.2 Train / Early-Stop / Test

用户当前要求：Q3 只训练一个 mixed 模型，early stopping 使用 mixed validation loss，最终分别在 real 和 generated test 上画图。

实现时必须明确记录：

- `mixed_train_manifest.csv`
- `mixed_val_manifest.csv`
- `real_test_manifest.csv`
- `generated_test_manifest.csv`

推荐 early stopping 指标：

- 默认使用 `mixed_val_manifest.csv` 上的 validation loss。
- 最终画图分别读取 real/generated test prediction。

注意：

- real/generated test 不参与 early stopping，只用于最终评估和画图。

### 7.3 输出目录建议

```text
outputs/q3_mixed_matched/
├── manifests/
├── checkpoints/
├── logs/
├── predictions/
└── figures/
```

## 8. Q4: Mixed + Peak-Valley Balanced Distribution

### 8.1 数据构建

Q4 是最接近 augmentation 生产场景的实验。

要求：

- mixed train、mixed val、mixed test 总量必须与 separated 实验对齐。
- 对每个 `age_bin × sex` 组合设置 fixed target。
- real 超过 target 中 real_part 时下采样，这是削峰。
- real 不足时全部保留，并用 generated 补齐到 target，这是填谷。
- generated 的目标是补足 real 稀疏组合，而不是模拟 real 分布。

### 8.2 Train / Early-Stop / Test

同 Q3：

- 只训练一个 mixed 模型。
- early stopping 使用 mixed validation loss。
- 分别保存 real test 和 generated test 的预测结果。
- 画图时画 Real Test 和 Generated Test 两条曲线。

### 8.3 输出目录建议

```text
outputs/q4_mixed_peak_valley/
├── manifests/
├── checkpoints/
├── logs/
├── predictions/
└── figures/
```

## 9. 图表要求

四种实验都需要输出按性别拆分的年龄段 MAE 图。

基本图形要求：

- 横轴：`Age Bin`
- 左轴：`MAE`
- 右轴：`Sample Count`
- 图例必须是论文风格 Title Case 英文。
- 图例不能遮挡曲线、柱子或坐标轴。
- 同时保存 PNG 和 PDF。

统一图形语法：

- Real 一律使用绿色。
- Generated 一律使用橙红色。
- MAE 只用线表示，不画 train MAE。
- Count 用柱表示。
- 非混合实验中，训练域的 count 柱必须堆叠为 `Train Count + Validation Count`；测试域 count 是单段柱。
- 混合实验中，mixed train count 必须堆叠为 `Real Train Count + Generated Train Count`，用于显示 augmentation 构成。
- 混合实验的 evaluation/test count 仍按 Real 和 Generated 分开显示。

推荐图注说明：

```text
Bars indicate sample counts. In separated-training settings, training-domain bars stack train and validation counts. In mixed-training settings, stacked bars decompose the training set into real and generated samples, while lines report MAE on real and generated evaluation sets.
```

建议每个象限至少输出：

- `male_age_bin_mae.png/pdf`
- `female_age_bin_mae.png/pdf`
- `age_bin_metrics.csv`
- `split_summary.csv`
- `source_age_sex_count_summary.csv`

## 10. 状态扫描与避免重复运行

新增或修改运行脚本时，必须支持扫描已有文件，避免重复运行已经完成的步骤。

推荐提供：

```text
scan_status.py
```

该脚本至少检查：

- manifest 是否存在
- checkpoint 是否存在
- prediction CSV 是否存在
- figure PNG/PDF 是否存在
- 每个象限的 completed/missing 状态

运行脚本应优先参考 scan 结果：

- 已有完整 checkpoint 时，不应默认重训。
- 已有 prediction CSV 时，不应默认重复推理。
- 已有 figure 时，不应默认重复画图，除非显式传入 force。

## 11. 当前已有结果处理

当前实验结果统一写入四象限目录：

- `outputs/q1_separated_matched/`
- `outputs/q2_separated_gen_balanced/`
- `outputs/q3_mixed_matched/`
- `outputs/q4_mixed_peak_valley/`

## 12. 验收标准

完成后应满足：

- 四个象限目录清晰存在。
- 每个象限的抽样策略可由 manifest summary 复核。
- generated-balanced 象限中，generated 的每个 `age_bin × sex` 数量一致或有明确不足说明。
- matched 象限中，real/generated 的 `age_bin × sex` 直方图一致或有明确不足说明。
- mixed 象限中，mixed train 的 real/generated 组成比例可追踪。
- 所有图例和轴标签达到论文可用标准。
- scan 脚本能报告哪些步骤已完成，避免重复跑长任务。
