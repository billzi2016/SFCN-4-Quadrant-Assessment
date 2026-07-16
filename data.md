# SFCN 数据组成说明

本文档固定说明 6 个最终实验的数据组成、数量计算和每个 `age_bin × gender` 组合的抽样规则。当前数值来自最新生成的 manifests。

## 总量表

| 项目 | 数值 | 计算方式 | 说明 |
| --- | ---: | --- | --- |
| Real 原始可用总量 | `10,364` | 从 `/Volumes/LuZhang16T/IU_Datasets` 收集 | 原始 real 数据池 |
| Generated 原始可用总量 | `41,410` | 从 `/Volumes/LuZhang16T/generated_mri` 收集 | 原始 generated 数据池 |
| Matched 基准总量 | `10,189` | 每个 `age_bin × gender` 取 `min(real, generated)` 后求和 | Q1/Q3 使用这个基准 |
| Generated 目标总量 | `10,189` | `= Matched 基准总量` | Q2/Q4 不允许 generated 比 real 多 |
| 年龄 bin 数 | `20` | `(105 - 5) / 5 = 20` | 每 5 岁一个 bin |
| 性别数 | `2` | `M / F` | 只保留男女两类 |
| age_bin × gender 组合数 | `40` | `20 × 2 = 40` | 所有均衡都按这个组合做 |
| Separated train 总量 | `8,150` | 分层 4:1 后实际结果 | Q1/Q2 单来源训练规模 |
| Separated val 总量 | `2,039` | 分层 4:1 后实际结果 | Q1/Q2 单来源验证规模 |
| Separated test 总量 | `10,189` | `= Matched 基准总量` | Q1/Q2 测试规模 |
| Balanced 每组目标 | `254 或 255` | `10,189 / 40 = 254 余 29` | 11 个组合为 254，29 个组合为 255 |
| Mixed train 总量 | `8,150` | 与 separated train 对齐 | Q3/Q4 训练规模 |
| Mixed val 总量 | `2,039` | 与 separated val 对齐 | Q3/Q4 验证规模 |
| Mixed test 总量 | `10,189` | 与 separated test 对齐 | Q3/Q4 测试规模 |

## 6 个实验的 Train / Val / Test 组成表

| 编号 | 实验 | Train 组成 | Val 组成 | Test 组成 |
| --- | --- | --- | --- | --- |
| 1 | Q1 Real -> Generated | Real matched train = `8,150` | Real matched val = `2,039` | Generated matched test = `10,189` |
| 2 | Q1 Generated -> Real | Generated matched train = `8,150` | Generated matched val = `2,039` | Real matched test = `10,189` |
| 3 | Q2 Real -> Generated-Balanced | Real matched train = `8,150` | Real matched val = `2,039` | Generated-balanced test = `10,189` |
| 4 | Q2 Generated-Balanced -> Real | Generated-balanced train = `8,149` | Generated-balanced val = `2,040` | Real matched test = `10,189` |
| 5 | Q3 Mixed Matched | Real matched train = `4,075` + Generated matched train = `4,075` | Real matched val = `1,019` + Generated matched val = `1,020` | Real matched test = `5,095` + Generated matched test = `5,094` |
| 6 | Q4 Mixed Peak-Valley Balanced | Real peak-valley train = `2,090` + Generated fill train = `6,060` | Real peak-valley val = `521` + Generated fill val = `1,518` | Real peak-valley test = `2,754` + Generated fill test = `7,435` |

## 核心原则

- 所有实验都控制 train、val、test 总量一致或尽量一致。
- Separated 实验是单来源训练：train 只来自 Real 或只来自 Generated。
- Mixed 实验不是把 Real 和 Generated 简单相加到 2 倍，而是让 mixed 总量等于 separated 总量。
- Q3 mixed 中 Real 和 Generated 尽量各占一半；因为 `10,189` 是奇数，val/test 会有 1 个样本差异。
- Q4 mixed 执行削峰填谷：Real 多的 age/gender 组合下采样，Real 少的组合用 Generated 补齐。
- Q1/Q3 使用 matched 分布：Generated 按 Real 的 `age_bin × gender` 直方图抽样。
- Q2 使用 generated-balanced 分布：Generated 自身在 40 个组合上均衡。
- 随机抽样使用 `seed = 42`，余数分配和下采样都必须可复现。

## 254/255 的来源

Balanced 分布要求把 `10,189` 个样本平均放到 `40` 个 `age_bin × gender` 组合：

```text
10,189 / 40 = 254 余 29
```

因此完整 balanced 分布是：

```text
11 个组合 × 254 = 2,794
29 个组合 × 255 = 7,395
总计 = 10,189
```

哪 29 个组合是 `255`，由 `seed = 42` 固定决定。

## 分布类型

| 分布 | Real 每个组合 | Generated 每个组合 | 总量控制 |
| --- | --- | --- | --- |
| Matched | 使用 matched 后 Real 分布 | 等于 Real 对应组合数量 | Real = Generated = `10,189` |
| Generated-Balanced | 不用于 Real | 11 组 254，29 组 255 | Generated = `10,189` |
| Peak-Valley Mixed | 超过目标则下采样，不足则保留 | 补齐 Real 不足的部分 | Mixed 总量 = `10,189` |

Matched 匹配的是 `age_bin × gender` 联合分布，也就是年龄分布和性别分布同时与 Real 对齐。

Balanced 平均的是 `age_bin × gender` 联合组合，也就是每个年龄段、每个性别组合都尽量一样多。

Peak-Valley Mixed 是“削峰填谷”：对 mixed 后的总分布做控制，Real 高峰下采样，Real 低谷用 Generated 补齐。

## 1. Q1 Real -> Generated

目标：只用 Real 训练，测试 Generated；Generated 的分布必须匹配 Real。

计算：

```text
Real matched selected = 10,189
Generated matched selected = 10,189
generated_count(age_bin, gender) = real_count(age_bin, gender)
```

Split：

```text
train = Real matched selected 分层 80% = 8,150
val = Real matched selected 分层 20% = 2,039
test = Generated matched selected = 10,189
```

训练模型数：

```text
1
```

## 2. Q1 Generated -> Real

目标：只用 Generated 训练，测试 Real；Generated 的分布仍然匹配 Real。

计算：

```text
Generated matched selected = 10,189
Real matched selected = 10,189
generated_count(age_bin, gender) = real_count(age_bin, gender)
```

Split：

```text
train = Generated matched selected 分层 80% = 8,150
val = Generated matched selected 分层 20% = 2,039
test = Real matched selected = 10,189
```

训练模型数：

```text
1
```

## 3. Q2 Real -> Generated-Balanced

目标：只用 Real 训练，测试均衡 Generated；训练总量和 Q1 保持一致。

计算：

```text
Real matched selected = 10,189
Generated balanced selected = 10,189
```

每个组合：

```text
Real: 使用 matched 后真实分布，不均衡
Generated: 11 个组合为 254，29 个组合为 255
```

Split：

```text
train = Real matched selected 分层 80% = 8,150
val = Real matched selected 分层 20% = 2,039
test = Generated balanced selected = 10,189
```

训练模型数：

```text
1
```

## 4. Q2 Generated-Balanced -> Real

目标：只用均衡 Generated 训练，测试 Real。

计算：

```text
Generated balanced selected = 10,189
Real matched selected = 10,189
```

每个组合：

```text
Generated: 11 个组合为 254，29 个组合为 255
Real: 使用 matched 后真实分布，不均衡
```

Split：

```text
train = Generated balanced selected 分层 80% = 8,149
val = Generated balanced selected 分层 20% = 2,040
test = Real matched selected = 10,189
```

训练模型数：

```text
1
```

## 5. Q3 Mixed Matched

目标：Real 和 Generated 混合训练，但总训练量不变；Generated 分布匹配 Real。

Q3 不是两个模型，只训练一个 mixed 模型。

先得到 Q1 matched 数据：

```text
Real matched pool = 10,189
Generated matched pool = 10,189
```

为了让 mixed train 与 separated train 一样大，mixed 中 Real 和 Generated 尽量各取一半：

```text
mixed_train_total = 8,150
real_train_part = 4,075
generated_train_part = 4,075
```

Val：

```text
mixed_val_total = 2,039
real_val_part = 1,019
generated_val_part = 1,020
```

Test：

```text
mixed_test_total = 10,189
real_test_part = 5,095
generated_test_part = 5,094
```

画图：

```text
Real Test 一条曲线
Generated Test 一条曲线
```

训练模型数：

```text
1
```

## 6. Q4 Mixed Peak-Valley Balanced

目标：Mixed 训练中执行削峰填谷，同时保持 train、val、test 总量与其他实验一致。

Q4 只训练一个 mixed 模型。

完整 balanced 目标：

```text
mixed_total = 10,189
每个 age_bin × gender 组合目标 = 254 或 255
```

Train 目标：

```text
mixed_train_total = 8,150
8,150 / 40 = 203 余 30
10 个组合为 203
30 个组合为 204
```

Val 目标：

```text
mixed_val_total = 2,039
2,039 / 40 = 50 余 39
1 个组合为 50
39 个组合为 51
```

Test 目标：

```text
mixed_test_total = 10,189
10,189 / 40 = 254 余 29
11 个组合为 254
29 个组合为 255
```

每个组合内执行削峰填谷：

```text
目标 real_part ≈ 组合目标 / 2
目标 generated_part = 组合目标 - real_part
```

如果 Real 超过 `real_part`：

```text
削峰：Real 下采样到 real_part
Generated 抽 generated_part
```

如果 Real 不足 `real_part`：

```text
填谷：Real 全部保留
Generated 抽 组合目标 - Real 数量
```

当前生成结果：

```text
train = Real 2,090 + Generated 6,060 = 8,150
val = Real 521 + Generated 1,518 = 2,039
test = Real 2,754 + Generated 7,435 = 10,189
```

画图：

```text
Real Test 一条曲线
Generated Test 一条曲线
```

训练模型数：

```text
1
```

## 最终模型数

| 象限 | 实验 | 模型数 |
| --- | --- | ---: |
| Q1 | Real -> Generated；Generated -> Real | 2 |
| Q2 | Real -> Generated-Balanced；Generated-Balanced -> Real | 2 |
| Q3 | Mixed Matched | 1 |
| Q4 | Mixed Peak-Valley Balanced | 1 |
| 总计 | 6 个实验 | 6 |

## 最终规模对齐

| 实验类型 | Train | Val | Test |
| --- | ---: | ---: | ---: |
| Separated | `8,149-8,150` | `2,039-2,040` | `10,189` |
| Mixed | `8,150` | `2,039` | `10,189` |

这样设计后，审稿人看到的是：

- 所有模型训练集大小基本一致。
- 所有模型验证集大小基本一致。
- 所有模型测试集大小一致。
- 差异只来自数据来源和分布策略，而不是样本量。
