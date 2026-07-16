# SFCN 四象限实验报告

本文件用于集中查看 SFCN 四象限实验结果。对应实验尚未完成时，部分图片可能暂时不存在；这里仍保留最终路径，等训练、推理和画图完成后会自动成为完整报告。

## 四象限定义

| 象限 | 训练设置 | 年龄分布设置 | 实验目的 |
| --- | --- | --- | --- |
| Q1 | 分开训练 | Real 与 Generated 年龄分布匹配 | 控制变量的实验室设置：real 和 generated 具有匹配的年龄/性别分布。 |
| Q2 | 分开训练 | Generated 年龄分布均衡 | 测试 generated 数据在各年龄段数量均衡时，跨域泛化表现如何。 |
| Q3 | 混合训练 | Real 与 Generated 年龄分布匹配 | 理想 augmentation 设置：real/generated 分布匹配后混合训练。 |
| Q4 | 混合训练 | 削峰填谷均衡 | 更接近生产 augmentation 的设置：real 高峰下采样，real 低谷用 generated 补齐。 |

## 图例说明

- Real 数据统一使用绿色。
- Generated 数据统一使用橙红色。
- 折线表示 MAE。
- 柱状图表示样本数量。
- 分开训练实验中，训练域柱状图堆叠 `Train + Validation` 数量。
- 混合训练实验中，训练柱状图堆叠 `Real Train + Generated Train` 数量。

## Q1：分开训练 + 年龄分布匹配

### 1. Q1 Real 训练，Generated 测试

![Q1 Real to Generated Male](outputs/q1_separated_matched/real-gen/figures/male_age_bin_mae.png)

男性图：`outputs/q1_separated_matched/real-gen/figures/male_age_bin_mae.png`

女性图：`outputs/q1_separated_matched/real-gen/figures/female_age_bin_mae.png`

### 2. Q1 Generated 训练，Real 测试

![Q1 Generated to Real Male](outputs/q1_separated_matched/gen-real/figures/male_age_bin_mae.png)

男性图：`outputs/q1_separated_matched/gen-real/figures/male_age_bin_mae.png`

女性图：`outputs/q1_separated_matched/gen-real/figures/female_age_bin_mae.png`

## Q2：分开训练 + Generated 年龄分布均衡

### 3. Q2 Real 训练，Generated-Balanced 测试

![Q2 Real to Generated-Balanced Male](outputs/q2_separated_gen_balanced/real-gen/figures/male_age_bin_mae.png)

男性图：`outputs/q2_separated_gen_balanced/real-gen/figures/male_age_bin_mae.png`

女性图：`outputs/q2_separated_gen_balanced/real-gen/figures/female_age_bin_mae.png`

### 4. Q2 Generated-Balanced 训练，Real 测试

![Q2 Generated-Balanced to Real Male](outputs/q2_separated_gen_balanced/gen-real/figures/male_age_bin_mae.png)

男性图：`outputs/q2_separated_gen_balanced/gen-real/figures/male_age_bin_mae.png`

女性图：`outputs/q2_separated_gen_balanced/gen-real/figures/female_age_bin_mae.png`

## Q3：混合训练 + 年龄分布匹配

### 5. Q3 Mixed Matched

![Q3 Mixed Matched Male](outputs/q3_mixed_matched/mixed/figures/male_age_bin_mae.png)

男性图：`outputs/q3_mixed_matched/mixed/figures/male_age_bin_mae.png`

女性图：`outputs/q3_mixed_matched/mixed/figures/female_age_bin_mae.png`

## Q4：混合训练 + 削峰填谷均衡

### 6. Q4 Mixed Peak-Valley Balanced

![Q4 Mixed Peak-Valley Balanced Male](outputs/q4_mixed_peak_valley/mixed/figures/male_age_bin_mae.png)

男性图：`outputs/q4_mixed_peak_valley/mixed/figures/male_age_bin_mae.png`

女性图：`outputs/q4_mixed_peak_valley/mixed/figures/female_age_bin_mae.png`

## 状态检查命令

```bash
python3 main.py scan-status --only-missing
```
