# PRD: Balanced Real-Generated Cross-Domain SFCN Experiments

## 1. 背景

`SFCN` 目录下旧方案是 preliminary 结果，核心设定是：

- 用全部 `real MRI` 训练
- 用少量 `generated MRI` 做 validation
- 用全量 `generated MRI` 做 test

这个设计不再适合当前实验目标。现在要验证的是：

- 真实 MRI 训练出的年龄段分类模型，能否泛化到 generated MRI
- generated MRI 训练出的年龄段分类模型，能否泛化到真实 MRI
- 两个方向的 train/val/test 数据口径必须对称、可复现、可比较

因此当前正式方案不再沿用旧的 `real_train + generated_val/test` 设计，而是重做两个方向的 cross-domain 实验。

## 2. 总目标

训练两个独立的 `SFCN` 年龄段分类模型：

1. `real-gen`
   - `real` 数据按 `4:1` 切分为 train 和 validation
   - `generated` 数据作为 test
   - 画图时比较 `real_val` 和 `generated_test`

2. `gen-real`
   - `generated` 数据按 `4:1` 切分为 train 和 validation
   - `real` 数据作为 test
   - 画图时比较 `generated_val` 和 `real_test`

两个实验必须使用相同的数据选择原则：

- `real` 和 `generated` 的样本数量必须相同
- `generated` 的年龄分布必须尽量均匀
- 所有随机筛选必须固定 `seed = 42`
- train/validation 切分必须按年龄均等切分，避免某些年龄段只出现在 train 或 validation

## 3. 非目标

本 PRD 不要求：

- 继续复用旧 preliminary 结果
- 使用旧的 generated validation 规则
- 用 generated 做 real 模型的 early stopping validation
- 用 real train 结果和 generated test 结果直接画在一起
- 改成连续年龄回归
- 使用外部预训练权重
- 修改 MRI header 或依赖 generated 的 NIfTI header 信息

## 4. 数据来源

### 4.1 Real MRI

- 根目录：`/Volumes/LuZhang16T/IU_Datasets`
- 标签文件：`/Volumes/LuZhang16T/IU_Datasets/mapping_table.csv`

标签字段至少包括：

- `dataset`
- `id`
- `dir`
- `sex`
- `age`

real 的年龄标签必须来自 `mapping_table.csv` 的 `age` 字段，不允许从文件名推断。

### 4.2 Generated MRI

- 根目录：`/Volumes/LuZhang16T/generated_mri`

generated 文件名解析规则继续使用当前统一解析器：

- `age1.00_sexM_s131.nii.gz`
- `0004_age0.00_sexF_s4.nii.gz`

规则：

- 可选前缀如 `0004_` 没有语义，忽略
- `ageX.XX` 是扩散模型条件年龄
- `sexM` / `sexF` 是性别
- `s131` / `s4` 是 sample id

generated 的 `age0.00 ~ age1.00` 暂按当前规则线性映射到 `5 ~ 105` 岁。该规则必须集中保留在 `common.py` 中，不允许散落在训练、评估或画图脚本里。

## 5. 数据预处理假设

当前继续沿用 v1 已确认的数据前提：

- `real` 和 `generated` 数组 shape 都是 `(182, 218, 182)`
- 数据已经完成 top/bottom percentile 裁剪和 z-score 标准化
- 训练脚本不重复做新的逐样本 z-score
- generated 的 NIfTI header 不可信
- 模型输入只按数组张量处理，不依赖 affine、spacing、origin 或 orientation

## 6. 年龄标签与任务定义

任务仍然是年龄段分类，不是连续年龄回归。

默认年龄范围：

- `5` 到 `105`

默认年龄段：

- `5-10`
- `10-15`
- `15-20`
- ...
- `100-105`

规则：

- 年龄段为左闭右开
- 最后一个年龄段包含右端点
- 模型输出为每个年龄段一个 logit
- 损失函数使用 `CrossEntropyLoss`
- 评估时将预测类别映射为年龄段中心值，再计算 MAE/MSE

## 7. 样本数量对齐原则

当前方案必须先构建一个数量对齐的数据基础：

- `N_generated_selected = generated` 年龄均衡抽样后的样本数
- `N_real_selected = N_generated_selected`

也就是说：

- generated 先按年龄均衡原则抽样
- real 再按同样总数抽样
- 最终 real 和 generated 样本数量必须完全一致

如果 real 可用数量少于目标 generated 数量，则必须降低 generated 抽样数量，让两者一致。

如果 generated 某些年龄样本不足，不能随意补 mock 数据，也不能复制样本。必须按真实可用样本重新确定每个年龄的可抽数量。

## 8. Generated 年龄均衡抽样

generated 的抽样必须按年龄均匀分布。

基本原则：

- 先按 `age_year` 分组
- 每个年龄尽量抽相同数量
- 每个年龄内使用随机种子 `42` 随机打乱后抽样
- 如果某些年龄样本不足，则该年龄只能抽可用数量
- 最终 generated selected set 应尽量覆盖完整年龄轴

建议实现：

1. 收集全部 generated 样本并解析出 `age_year`、`sex`、`sample_id`
2. 按 `age_year` 分组
3. 计算每个年龄可用数量
4. 取所有年龄可用数量的最小值作为基础 `per_age_quota`
5. 如果 `per_age_quota` 过小，需要在日志中明确提示
6. 对每个年龄组用 `seed = 42` 打乱
7. 每个年龄抽 `per_age_quota` 个样本
8. 合并后得到 `generated_selected_manifest.csv`

性别不作为强制均衡条件，但必须在 manifest summary 中统计每个年龄内的 `M/F` 数量。若某些年龄性别严重偏斜，必须在日志中显示，不能静默忽略。

## 9. Real 数量对齐抽样

real 的抽样目标是：

- 总数等于 `generated_selected_manifest.csv`
- 年龄分布尽量与 generated selected set 对齐
- 使用随机种子 `42`

建议实现：

1. 构建全部可用 real manifest
2. 按 `age_year` 或 `age_bin` 分组
3. 优先按 generated selected 的年龄计数抽取 real
4. 如果 real 的某个具体整数年龄不足，则允许降级到对应 `5-year age_bin` 内补足
5. 所有补足逻辑必须记录在 summary 中
6. 最终输出 `real_selected_manifest.csv`

必须保证：

- `len(real_selected_manifest) == len(generated_selected_manifest)`
- 不允许重复 real 样本
- 不允许使用没有真实标签的 real 样本

## 10. 年龄与性别均衡 train/validation 切分

train/validation 切分比例为 `4:1`。

切分要求：

- 只对当前训练域的数据做 train/validation 切分
- `real-gen` 中只切 real selected
- `gen-real` 中只切 generated selected
- test 域不参与 train/validation 切分
- 切分必须按 `age_bin × gender` 分层后执行
- 每个 `age_bin × gender` 组合在 train 和 validation 中都要尽量保持均衡
- 使用 `seed = 42`

推荐切分规则：

1. 对训练域 selected manifest 按 `age_bin × sex` 分组，例如 `5-10/M`、`5-10/F`、`10-15/M`、`10-15/F`
2. 每个组合内使用 `seed = 42` 随机打乱
3. 每个组合按 `80% train / 20% validation` 切分
4. 同一个 `age_bin` 内，`M` 和 `F` 的数量应先按较小可用数量截断到一致，再切分 train/validation
5. 如果某个 `age_bin × sex` 组合样本不足，不能让另一个性别直接占满该 age_bin；必须记录该组合不足，并按该 age_bin 内可实现的最小平衡数量处理
6. 如果某个 age_bin 中任一性别完全缺失，则该 age_bin 不能作为严格性别均衡训练/验证单元，必须在 summary 中标记为 excluded 或 imbalance-risk
7. 如果某个组合样本过少，至少保证 validation 中有样本；若无法保证，必须在日志和 summary 中报告

这里的核心不是只保证年龄均匀，而是保证 `age_bin × gender` 四类组合口径稳定。对于任意相邻或对照年龄段，不能出现某个年龄段几乎全是男性、另一个年龄段几乎全是女性的情况，否则模型可能学到性别偏差而不是年龄信息。

严格要求：

- train 中每个 `age_bin` 的 `M/F` 数量应一致或尽量一致
- validation 中每个 `age_bin` 的 `M/F` 数量应一致或尽量一致
- train 和 validation 的 `age_bin × sex` 组合数量必须写入 summary
- 如果为了平衡丢弃了样本，必须记录被丢弃数量
- 不允许为了补齐组合而复制样本
- 不允许使用 mock 样本补齐组合

## 11. 实验 A: real-gen

### 11.1 数据定义

`real-gen` 代表：

- train domain: `real`
- validation domain: `real`
- test domain: `generated`

数据文件：

- `SFCN/outputs/real-gen/manifests/real_selected_manifest.csv`
- `SFCN/outputs/real-gen/manifests/generated_selected_manifest.csv`
- `SFCN/outputs/real-gen/manifests/real_train_manifest.csv`
- `SFCN/outputs/real-gen/manifests/real_val_manifest.csv`
- `SFCN/outputs/real-gen/manifests/generated_test_manifest.csv`
- `SFCN/outputs/real-gen/manifests/split_summary.csv`

其中：

- `real_train_manifest.csv` 来自 real selected 的 80%
- `real_val_manifest.csv` 来自 real selected 的 20%
- `generated_test_manifest.csv` 应等于 generated selected 全部样本

### 11.2 训练

模型只使用 `real_train_manifest.csv` 训练。

early stopping 只使用 `real_val_manifest.csv`。

最佳 checkpoint 保存到：

- `SFCN/outputs/real-gen/checkpoints/best_model.pt`

训练日志保存到：

- `SFCN/outputs/real-gen/logs/train_log.csv`
- `SFCN/outputs/real-gen/logs/training_config.json`

### 11.3 推理与评估

训练完成后必须分别推理：

- `real_val_manifest.csv`
- `generated_test_manifest.csv`

预测结果：

- `SFCN/outputs/real-gen/predictions/real_val_predictions.csv`
- `SFCN/outputs/real-gen/predictions/generated_test_predictions.csv`

画图和表格必须比较：

- `real_val`
- `generated_test`

不允许把 `real_train` 放进这张图。

## 12. 实验 B: gen-real

### 12.1 数据定义

`gen-real` 代表：

- train domain: `generated`
- validation domain: `generated`
- test domain: `real`

数据文件：

- `SFCN/outputs/gen-real/manifests/generated_selected_manifest.csv`
- `SFCN/outputs/gen-real/manifests/real_selected_manifest.csv`
- `SFCN/outputs/gen-real/manifests/generated_train_manifest.csv`
- `SFCN/outputs/gen-real/manifests/generated_val_manifest.csv`
- `SFCN/outputs/gen-real/manifests/real_test_manifest.csv`
- `SFCN/outputs/gen-real/manifests/split_summary.csv`

其中：

- `generated_train_manifest.csv` 来自 generated selected 的 80%
- `generated_val_manifest.csv` 来自 generated selected 的 20%
- `real_test_manifest.csv` 应等于 real selected 全部样本

### 12.2 训练

模型只使用 `generated_train_manifest.csv` 训练。

early stopping 只使用 `generated_val_manifest.csv`。

最佳 checkpoint 保存到：

- `SFCN/outputs/gen-real/checkpoints/best_model.pt`

训练日志保存到：

- `SFCN/outputs/gen-real/logs/train_log.csv`
- `SFCN/outputs/gen-real/logs/training_config.json`

### 12.3 推理与评估

训练完成后必须分别推理：

- `generated_val_manifest.csv`
- `real_test_manifest.csv`

预测结果：

- `SFCN/outputs/gen-real/predictions/generated_val_predictions.csv`
- `SFCN/outputs/gen-real/predictions/real_test_predictions.csv`

画图和表格必须比较：

- `generated_val`
- `real_test`

不允许把 `generated_train` 放进这张图。

## 13. 输出目录结构

当前输出目录固定为：

```text
SFCN/outputs/
├── real-gen/
│   ├── manifests/
│   ├── checkpoints/
│   ├── logs/
│   ├── predictions/
│   └── figures/
└── gen-real/
    ├── manifests/
    ├── checkpoints/
    ├── logs/
    ├── predictions/
    └── figures/
```

旧的 v1 输出目录不再使用：

- `SFCN/outputs/manifests/`
- `SFCN/outputs/checkpoints/`
- `SFCN/outputs/logs/`
- `SFCN/outputs/predictions/`
- `SFCN/outputs/figures/`

## 14. Manifest 字段要求

所有 manifest 至少包含：

- `path`
- `sex`
- `age_year`
- `age_raw`
- `age_bin`
- `age_bin_label`
- `source`

real 额外建议包含：

- `dataset`
- `subject_id`

generated 额外建议包含：

- `sample_id`

所有 manifest 必须可复现。只要输入数据不变、`seed=42` 不变，输出 manifest 行集合和顺序都应一致。

## 15. Summary 与审计文件

每个实验目录必须输出：

- `split_summary.csv`
- `age_bin_count_summary.csv`
- `sex_age_count_summary.csv`

summary 至少要能回答：

- real selected 总数是多少
- generated selected 总数是多少
- train/val/test 各有多少
- 每个 age_bin 的 train/val/test 数量是多少
- 每个 source、sex、age_bin 的数量是多少
- 哪些年龄或 age_bin 因样本不足发生了降级或补足

如果出现无法数量对齐、年龄缺失、某个 age_bin 过少等问题，脚本必须明确报错或写入 summary，不能静默继续。

## 16. 训练配置

默认训练配置可以沿用 v1：

- `batch_size = 4`
- `num_workers = 4`
- `max_epochs = 50`
- `patience = 3`
- `lr = 2e-4`
- `weight_decay = 1e-4`
- `device = mps`
- `seed = 42`

但配置文件必须分别保存到两个实验目录，不能共用。

## 17. 评估指标

逐样本 prediction CSV 至少包含：

- `path`
- `sex`
- `age_year`
- `age_bin`
- `age_bin_label`
- `target_class`
- `pred_class`
- `pred_prob`
- `pred_age_center`
- `absolute_error`
- `squared_error`
- `source_label`
- `experiment`

聚合指标至少包含：

- classification accuracy
- MAE
- MSE
- 按 sex 的 MAE/MSE
- 按 age_bin 的 MAE/MSE
- 按 sex + age_bin 的 MAE/MSE

## 18. 画图要求

每个实验分别画图，不混在一起。

### 18.1 real-gen 图

图中只比较：

- `real_val`
- `generated_test`

输出：

- `SFCN/outputs/real-gen/figures/male_age_bin_mae.png`
- `SFCN/outputs/real-gen/figures/male_age_bin_mae.pdf`
- `SFCN/outputs/real-gen/figures/female_age_bin_mae.png`
- `SFCN/outputs/real-gen/figures/female_age_bin_mae.pdf`

### 18.2 gen-real 图

图中只比较：

- `generated_val`
- `real_test`

输出：

- `SFCN/outputs/gen-real/figures/male_age_bin_mae.png`
- `SFCN/outputs/gen-real/figures/male_age_bin_mae.pdf`
- `SFCN/outputs/gen-real/figures/female_age_bin_mae.png`
- `SFCN/outputs/gen-real/figures/female_age_bin_mae.pdf`

### 18.3 图形内容

每张图建议包含：

- 横轴：`age_bin`
- 左轴：MAE 或 MSE
- 右轴：样本数 count
- 两条误差曲线分别代表 validation domain 和 cross-domain test
- 背景柱表示每个年龄段样本数

图例必须明确标注 source label，例如：

- `real_val`
- `generated_test`
- `generated_val`
- `real_test`

## 19. 推荐执行流程

推荐提供 6 个训练/推理单步 shell 入口：

- `SFCN/run_train_real_gen.sh`
- `SFCN/run_train_gen_real.sh`
- `SFCN/run_infer_real_gen_val.sh`
- `SFCN/run_infer_real_gen_test.sh`
- `SFCN/run_infer_gen_real_val.sh`
- `SFCN/run_infer_gen_real_test.sh`

另提供一个统一画图脚本：

- `SFCN/run_plot.sh`

同时保留：

- `SFCN/run_all.sh`

其中：

- 两个 train 脚本只负责训练，不自动推理或画图
- 四个 infer 脚本只负责单个 experiment/split 的推理
- `run_plot.sh` 统一画两个方向的图
- `run_all.sh` 是全流程命令清单，方便在 Linux/终端复制粘贴；可以直接执行，但不推荐在已知训练或推理可能卡住时一口气运行

拆分脚本的原因是训练和推理阶段会使用模型与数据加载，可能触发较重的线程库、OpenMP 或多进程 DataLoader 行为。画图阶段不使用 DataLoader 多进程，因此可以合并成一个脚本。任何一步卡住后都应该能单独重跑，不应强制从头跑完整流程。

完整流程依次执行：

1. 构建 manifests
2. 训练 `real-gen`
3. 推理 `real-gen` validation 和 test
4. 评估并画 `real-gen` 图
5. 训练 `gen-real`
6. 推理 `gen-real` validation 和 test
7. 评估并画 `gen-real` 图

训练参数默认沿用当前代码，不在这些 shell 脚本里额外改 batch size、learning rate、patience 等参数。

### 19.1 构建 manifests

使用脚本：

- `SFCN/build_manifests.py`

该脚本负责：

1. 收集全部 real
2. 收集全部 generated
3. 按年龄均衡抽 generated
4. 抽同等数量 real
5. 写出 `real-gen` manifests
6. 写出 `gen-real` manifests
7. 写出 summary

### 19.2 训练 real-gen

推荐命令形式：

```bash
python3 train.py --experiment real-gen --device mps --num-workers 4
```

该命令读取：

- `SFCN/outputs/real-gen/manifests/real_train_manifest.csv`
- `SFCN/outputs/real-gen/manifests/real_val_manifest.csv`

保存到：

- `SFCN/outputs/real-gen/checkpoints/`
- `SFCN/outputs/real-gen/logs/`

### 19.3 训练 gen-real

推荐命令形式：

```bash
python3 train.py --experiment gen-real --device mps --num-workers 4
```

该命令读取：

- `SFCN/outputs/gen-real/manifests/generated_train_manifest.csv`
- `SFCN/outputs/gen-real/manifests/generated_val_manifest.csv`

保存到：

- `SFCN/outputs/gen-real/checkpoints/`
- `SFCN/outputs/gen-real/logs/`

### 19.4 推理和画图

推荐新增统一推理脚本支持 `--experiment`：

```bash
python3 infer.py --experiment real-gen --split val
python3 infer.py --experiment real-gen --split test
python3 infer.py --experiment gen-real --split val
python3 infer.py --experiment gen-real --split test
```

推荐新增统一评估画图脚本：

```bash
python3 evaluate_plot.py --experiment real-gen
python3 evaluate_plot.py --experiment gen-real
```

## 20. 验收标准

当前方案完成后必须满足：

- `real-gen` 和 `gen-real` 两个目录都存在完整输出子目录
- real selected 和 generated selected 数量完全一致
- generated selected 年龄分布尽量均匀，并有 summary 可查
- 所有随机抽样都由 `seed=42` 控制
- `real-gen` 只用 real train/val 训练和 early stopping
- `real-gen` 的 test 只用 generated test
- `gen-real` 只用 generated train/val 训练和 early stopping
- `gen-real` 的 test 只用 real test
- 每个实验的图只比较自己的 validation 和 cross-domain test
- 不再依赖旧 preliminary 输出

## 21. 当前目录清理状态

旧结果目录已清理，当前目录结构已建立：

```text
SFCN/outputs/real-gen/
SFCN/outputs/gen-real/
```

后续实现代码时，应只向这两个实验目录写入新结果，避免覆盖或混用旧 v1 文件。
