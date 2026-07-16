# PRD: 在真实 MRI 上从零训练 Torch 版 SFCN，并用 generated MRI 做验证与测试

## 1. 背景

之前 `SFCN` 效果差，主要怀疑原因是模型并不是在当前项目使用的真实 MRI 数据分布上训练出来的，而是依赖外部数据或外部预训练权重。  
当前目标不是继续迁移旧模型，而是：

- 在当前真实 MRI 数据集上，从零训练一个 `PyTorch` 版 `SFCN`
- 不再把任务定义为连续年龄回归
- 改为年龄段分类任务
- 用少量 `generated MRI` 做 validation 和 early stopping
- 在训练结束后，对整个 `generated MRI` 集合做全量测试
- 将测试结果完整保存，画图逻辑单独拆出


## 2. 目标

- 用真实 MRI 数据训练一个新的 3D CNN 年龄分类模型
- 模型结构以 `SFCN` 为基础思想，但实现直接使用 `torch`
- 任务形式改为年龄段分类，而不是预测具体年龄
- 损失函数使用 `CrossEntropyLoss`
- 训练完成后对全量 generated MRI 做推理
- 输出按性别拆分的年龄段误差图
  - `Male` 一张
  - `Female` 一张
- 图的横轴为年龄段
- 图的纵轴为 `MSE`
- 测试阶段必须保留逐样本结果，不允许只保留最终图


## 3. 非目标

- 当前 PRD 不要求复用外部预训练模型
- 当前 PRD 不要求做连续 brain age regression
- 当前 PRD 不要求在本阶段引入复杂公平学习方法
- 当前 PRD 不要求依赖 generated MRI 的 header 空间信息
- 当前 PRD 不要求把画图逻辑和训练脚本强耦合


## 4. 数据来源

### 4.1 真实 MRI

- 路径：`/Volumes/LuZhang16T/IU_Datasets`
- 标签文件：`/Volumes/LuZhang16T/IU_Datasets/mapping_table.csv`
- 已确认标签字段至少包含：
  - `dataset`
  - `id`
  - `dir`
  - `sex`
  - `age`

真实 MRI 是训练主数据源。监督标签来自 `mapping_table.csv`，不从文件名反推。

### 4.2 Generated MRI

- 路径：`/Volumes/LuZhang16T/generated_mri`
- 当前仍在下载，尚未完整
- 最终预计规模：`40000+`
- 当前目标用途：
  - validation
  - 训练完成后的全量测试

generated MRI 不是训练主数据源，不参与主训练，只用于：

- 训练期间验证
- early stopping
- 训练结束后的全量推理与评估


## 5. 当前已确认的数据假设

### 5.1 张量尺寸

已抽样确认：

- `real` 体数据 shape 为 `(182, 218, 182)`
- `generated` 体数据 shape 也为 `(182, 218, 182)`

因此两类数据可以直接进入同一套 3D CNN 输入管线，不需要在训练脚本中额外做重采样到别的尺寸。

### 5.2 空间 header

已抽样确认：

- `real` 的 affine / spacing 是统一的标准空间风格
- `generated` 的 affine / spacing 不可信，不能当作真实物理空间信息使用

因此本项目明确规定：

- `generated MRI` 只按数组张量解释
- 不依赖其 `affine / spacing / origin / orientation`
- 不在训练、验证、测试时基于 generated header 做空间推断

换句话说，本项目默认：

- 可比性来自“数组空间已经统一”
- 而不是来自 NIfTI header 的物理空间一致性

### 5.3 强度归一化

当前数据已做过：

- 去掉 top 1% 像素点
- 去掉 bottom 1% 像素点
- z-score 标准化

抽样统计显示：

- `real` 与 `generated` 都已接近 `mean = 0, std = 1`
- 强度范围稳定

因此当前输入归一化方案是可接受的。  
训练脚本中不再重复做新的逐样本 z-score，只做：

- 读取
- dtype 转换
- tensor 化
- 必要的轻量数据增强


## 6. 任务定义

### 6.1 主任务

将原来的“连续年龄回归”改为“年龄段分类”。

原因：

- 年龄回归对长尾年龄分布更敏感
- 稀疏年龄段更容易被整体数据分布淹没
- 当前下游分析本来就是按年龄段汇总误差
- 分类目标与最终分析目标更一致

### 6.2 分类标签

按 `5` 岁为一个年龄段进行分桶。

建议默认年龄段：

- `5-10`
- `10-15`
- `15-20`
- ...
- `100-105`

实现时采用：

- 左闭右开区间
- 最后一个区间右端可闭合处理

真实年龄标签来自 `mapping_table.csv` 的 `age` 字段，映射到对应年龄段类别。

### 6.3 输出形式

模型输出：

- 每个年龄段一个 logit
- 通过 `softmax` 得到年龄段分类概率

训练损失：

- `CrossEntropyLoss`

### 6.4 为什么不用回归

当前阶段优先目标不是输出最精确的具体年龄，而是回答：

- generated MRI 是否保留足够的年龄信息
- 模型能否稳定区分不同年龄段
- 不同性别下，各年龄段误差如何变化

因此分类问题更合适。


## 7. 年龄公平性要求

模型不能只在高样本密度年龄段表现好，而在稀疏年龄段明显失真。  
因此“年龄公平”不是靠网络名字保证，而是靠训练策略保证。

本项目要求至少包含以下机制：

### 7.1 年龄段均衡采样

训练集按 `5-year bins` 分桶后，使用年龄段均衡采样策略，避免常见年龄段主导训练。

### 7.2 性别分布监控

训练日志中要同时记录：

- 每个年龄段样本数
- 每个年龄段内的男女数量

### 7.3 评估必须按年龄段拆开

不允许只汇报整体准确率或整体 loss。  
必须输出：

- 每个年龄段的误差
- 每个性别的年龄段误差图


## 8. Generated 命名规则

当前 generated MRI 至少存在两种命名形式：

- `age1.00_sexM_s131.nii.gz`
- `0004_age0.00_sexF_s4.nii.gz`

解析规则：

- 可选前缀如 `0004_` 无实际语义，直接忽略
- `ageX.XX` 为年龄条件字段
- `sexM` / `sexF` 为性别
- `s131` / `s4` 中的 `s` 表示 sample id

实现要求：

- 用统一解析器兼容这两种命名
- 前缀数字不参与任何标签逻辑

### 8.1 generated 年龄字段解释

当前约定中，generated 文件名中的 `age` 是扩散模型条件年龄字段。  
实现阶段需要统一把它映射到最终的年龄索引或年龄段。

默认约定：

- `age0.00` 到 `age1.00` 对应完整年龄轴
- 后续如果下载完成后的元数据定义更明确，则以最终元数据为准

PRD 要求实现代码将“年龄数值映射规则”写成单独函数，不要把规则散落在数据加载逻辑里。


## 9. 数据划分策略

### 9.1 训练集

- 来源：`real MRI`
- 标签：`mapping_table.csv`
- 用途：主训练

### 9.2 验证集

- 来源：`generated MRI`
- 按年龄与性别抽样
- 当前规则：
  - 每个年龄
  - 每个性别
  - 各抽 `10` 张

也就是理想情况下：

- `101` 个年龄
- `2` 个性别
- `10` 张/年龄/性别

共约：

- `2020` 张 validation 样本

### 9.3 Validation 抽样规则

当前实现目标直接固定为：

- 每个年龄
- 每个性别
- 精确取 `10` 张

如果某个年龄、某个性别不足 `10` 张：

- 直接报错
- 停止构建 validation manifest
- 不允许静默跳过
- 不允许自动退化成“有多少用多少”

这样做的原因是：

- validation 口径必须固定
- 否则不同下载阶段得到的验证集规模不同，结果不可比
- 当前你希望后续运行时直接使用固定规则，不再保留弹性分支

### 9.4 测试集

- 来源：完整 `generated MRI`
- 用途：训练结束后的全量推理

注意：

- validation 抽样子集与最终全量 generated 测试集是两个概念
- 训练结束后必须重新跑全量 generated


## 10. 模型方案

### 10.1 总体要求

模型用 `PyTorch` 直接实现，不依赖旧仓库实现。

### 10.2 网络要求

采用 `SFCN` 风格的 3D CNN：

- 多层 3D convolution block
- 下采样
- 最终 global pooling / flatten
- 全连接分类头

实现要求：

- 代码可读
- 不依赖魔法常量散落在脚本中
- 网络、训练、推理、评估、画图分文件组织

### 10.3 输入

- 单通道 3D MRI
- shape：`(1, 182, 218, 182)`

### 10.4 输出

- `num_bins` 维分类 logits


## 11. 损失函数与优化

### 11.1 损失函数

- `CrossEntropyLoss`

### 11.2 优化器

默认推荐：

- `AdamW`

默认首选训练参数：

- `batch_size = 4`
- `learning_rate = 2e-4`
- `weight_decay = 1e-4`

这样设置的原因是：

- 当前优先保证 MPS 训练稳定性，而不是把 batch 开得过大
- `batch=4` 比 `batch=8` 更稳，也比过小 batch 更有效率
- 学习率同步下调到更匹配 `batch=4` 的档位，降低训练初期震荡风险

### 11.3 训练监控

至少记录：

- training loss
- validation loss
- validation accuracy
- validation MSE
- 分性别 validation MSE
- 分年龄段 validation MSE

终端日志要求：

- 不使用 `tqdm` 之类的动态进度条
- 只使用简单文本进度输出
- 例如：`100/10922`
- 同时打印当前阶段的关键指标，保证日志可读但不过度花哨


## 12. Early Stopping

### 12.1 规则

- `patience = 3`

### 12.2 监控指标

early stopping 只监控：

- `validation accuracy`

说明：

- `patience = 3`
- 其余指标仍然全部打印和保存
- 但不用于 early stopping 判定


## 13. 训练完成后的测试要求

训练完成后，必须对完整 `generated MRI` 集合进行推理。

同时，为了做对比，也必须对 `real train` 集合再跑一次同一模型的推理。

输出至少包括：

- 每个样本的文件路径
- sample id
- sex
- 原始 age 条件字段
- 映射后的年龄
- 映射后的年龄段
- 真实类别索引或目标年龄段
- 预测类别索引
- 预测概率
- 若需要，可附带由类别中心恢复出的年龄代表值


## 14. 结果保存要求

测试结果必须保存成稳定、可复用的数据文件，不能只留图。

建议至少保存：

- `csv`
- 或 `parquet`

建议字段：

- `path`
- `sex`
- `sample_id`
- `age_raw`
- `age_year`
- `age_bin`
- `target_class`
- `pred_class`
- `pred_prob`
- `pred_age_center`
- `squared_error`

其中：

- `pred_age_center` 表示将预测年龄段映射为该年龄段中心值
- `squared_error` 用于后续按年龄段、按性别计算 `MSE`


## 15. 画图要求

画图代码必须单独写，不能和训练/测试主逻辑硬绑死。

原因：

- 后续图形样式、分桶方式、统计口径可能会反复修改
- 测试结果数据应保持稳定
- 画图逻辑应该是可反复迭代的后处理步骤

### 15.1 图输出要求

最终至少输出两张图：

- `Male`：横轴年龄段，纵轴 `MSE`
- `Female`：横轴年龄段，纵轴 `MSE`

并且每张图都要同时展示两条对比曲线：

- `generated`
- `real_train`

### 15.2 画图数据来源

画图必须直接读取测试阶段保存的结果文件，不重新跑模型。


## 16. 推荐目录结构

`SFCN` 目录建议后续组织为：

```text
SFCN/
  PRD_real_train_generated_val.md
  configs/
  data/
  models/
  train.py
  infer_generated.py
  build_validation_split.py
  evaluate_generated.py
  plot_age_bin_mse.py
  outputs/
    manifests/
    checkpoints/
    logs/
    predictions/
    figures/
```


## 17. 必须产出的中间文件

至少要有以下落盘产物：

- `real_train_manifest.csv`
- `generated_val_manifest.csv`
- `generated_test_manifest.csv`
- `best_model.pt`
- `train_log.csv`
- `generated_predictions.csv`
- `real_train_predictions.csv`
- `male_age_bin_mse.csv`
- `female_age_bin_mse.csv`

说明：

- `generated_predictions.csv` 与 `real_train_predictions.csv` 是画图和后续分析的主依据
- `male_age_bin_mse.csv` 与 `female_age_bin_mse.csv` 是汇总结果
- 图只是汇总结果的可视化，不是唯一结果


## 18. 风险与注意事项

### 18.1 generated 数据尚未下载完成

当前不能假设：

- 所有年龄都已完整到位
- 每个年龄每个性别都已有足够样本

因此实现必须支持渐进式执行。

### 18.2 generated 年龄字段映射需要固定规则

不能把年龄映射逻辑散落在多个脚本里。  
必须集中在单一解析函数中，避免后续统计口径不一致。

### 18.3 不依赖 generated header

generated 的 header 没有可信源信息，不能拿它做物理空间解释。

### 18.4 不能只看整体指标

如果只看整体 accuracy 或整体 loss，很可能掩盖某些年龄段的明显失败。  
必须按性别、按年龄段拆开看。


## 19. 结论

本项目将：

- 在 `real MRI` 上从零训练 `torch` 版 `SFCN`
- 将任务改为 `5-year` 年龄段分类
- 使用 `CrossEntropyLoss`
- 使用少量 `generated MRI` 做 validation 与 early stopping
- 用完整 `generated MRI` 做训练后全量测试
- 将逐样本结果完整保存
- 将画图逻辑独立成单独脚本
- 最终输出男女分开的年龄段 `MSE` 图

这个方案的核心不是复用旧模型，而是建立一条可解释、可复现、可反复分析的本地训练与评估流程。
