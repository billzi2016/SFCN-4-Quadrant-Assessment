# SFCN Data Composition

This document fixes the data composition, count calculations, and `age_bin x gender` sampling rules for the 6 final experiments. The current values come from the latest generated manifests.

## Totals

| Item | Value | Calculation | Notes |
| --- | ---: | --- | --- |
| Available real samples | `10,364` | Collected from `/Volumes/LuZhang16T/IU_Datasets` | Raw real-data pool |
| Available generated samples | `41,410` | Collected from `/Volumes/LuZhang16T/generated_mri` | Raw generated-data pool |
| Matched baseline total | `10,189` | Sum of `min(real, generated)` for each `age_bin x gender` group | Used by Q1/Q3 |
| Generated target total | `10,189` | `= Matched baseline total` | Q2/Q4 do not allow generated data to exceed real-data scale |
| Number of age bins | `20` | `(105 - 5) / 5 = 20` | One bin every 5 years |
| Number of genders | `2` | `M / F` | Only male and female are retained |
| `age_bin x gender` groups | `40` | `20 x 2 = 40` | All balancing is performed at this joint-group level |
| Separated train total | `8,150` | Actual result after stratified 4:1 split | Single-source train size for Q1/Q2 |
| Separated validation total | `2,039` | Actual result after stratified 4:1 split | Single-source validation size for Q1/Q2 |
| Separated test total | `10,189` | `= Matched baseline total` | Test size for Q1/Q2 |
| Balanced per-group target | `254 or 255` | `10,189 / 40 = 254 remainder 29` | 11 groups have 254, 29 groups have 255 |
| Mixed train total | `8,150` | Aligned with separated train | Train size for Q3/Q4 |
| Mixed validation total | `2,039` | Aligned with separated validation | Validation size for Q3/Q4 |
| Mixed test total | `10,189` | Aligned with separated test | Test size for Q3/Q4 |

## Train / Validation / Test Composition

| No. | Experiment | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| 1 | Q1 Real -> Generated | Real matched train = `8,150` | Real matched val = `2,039` | Generated matched test = `10,189` |
| 2 | Q1 Generated -> Real | Generated matched train = `8,150` | Generated matched val = `2,039` | Real matched test = `10,189` |
| 3 | Q2 Real -> Generated-Balanced | Real matched train = `8,150` | Real matched val = `2,039` | Generated-balanced test = `10,189` |
| 4 | Q2 Generated-Balanced -> Real | Generated-balanced train = `8,149` | Generated-balanced val = `2,040` | Real matched test = `10,189` |
| 5 | Q3 Mixed Matched | Real matched train = `4,075` + Generated matched train = `4,075` | Real matched val = `1,019` + Generated matched val = `1,020` | Real matched test = `5,095` + Generated matched test = `5,094` |
| 6 | Q4 Mixed Peak-Valley Balanced | Real peak-valley train = `2,090` + Generated fill train = `6,060` | Real peak-valley val = `521` + Generated fill val = `1,518` | Real peak-valley test = `2,754` + Generated fill test = `7,435` |

## Core Principles

- All experiments keep train, validation, and test totals identical or nearly identical.
- Separated experiments use single-source training: train data is either only real or only generated.
- Mixed experiments do not simply concatenate real and generated data into a doubled training set. Their mixed totals match the separated totals.
- In Q3 mixed training, real and generated samples are split as evenly as possible. Because `10,189` is odd, validation and test have a one-sample difference.
- Q4 mixed training applies peak-valley balancing: overrepresented real `age_bin x gender` groups are downsampled, and underrepresented real groups are filled with generated samples.
- Q1/Q3 use matched distributions: generated samples are drawn to match the real `age_bin x gender` histogram.
- Q2 uses a generated-balanced distribution: generated samples are balanced over the 40 joint groups.
- Random sampling uses `seed = 42`; remainder allocation and downsampling must be reproducible.

## Source of the 254/255 Targets

The balanced distribution puts `10,189` samples into `40` `age_bin x gender` groups:

```text
10,189 / 40 = 254 remainder 29
```

Therefore the full balanced distribution is:

```text
11 groups x 254 = 2,794
29 groups x 255 = 7,395
Total = 10,189
```

The 29 groups assigned `255` are fixed by `seed = 42`.

## Distribution Types

| Distribution | Real per group | Generated per group | Total control |
| --- | --- | --- | --- |
| Matched | Uses the matched real distribution | Equal to the corresponding real group count | Real = Generated = `10,189` |
| Generated-Balanced | Not used for real | 11 groups have 254, 29 groups have 255 | Generated = `10,189` |
| Peak-Valley Mixed | Downsample if above target, keep all if below target | Fill the missing real count | Mixed total = `10,189` |

Matched balancing matches the joint `age_bin x gender` distribution, so both age and gender distributions are aligned to real data.

Balanced sampling averages the joint `age_bin x gender` groups, meaning each age-bin and gender combination has as similar a count as possible.

Peak-Valley Mixed controls the final mixed distribution: real-data peaks are downsampled, and real-data valleys are filled with generated data.

## 1. Q1 Real -> Generated

Goal: train only on real data and test on generated data. The generated distribution must match the real distribution.

```text
Real matched selected = 10,189
Generated matched selected = 10,189
generated_count(age_bin, gender) = real_count(age_bin, gender)
```

Split:

```text
train = stratified 80% of Real matched selected = 8,150
val = stratified 20% of Real matched selected = 2,039
test = Generated matched selected = 10,189
```

Number of trained models:

```text
1
```

## 2. Q1 Generated -> Real

Goal: train only on generated data and test on real data. The generated distribution still matches real data.

```text
Generated matched selected = 10,189
Real matched selected = 10,189
generated_count(age_bin, gender) = real_count(age_bin, gender)
```

Split:

```text
train = stratified 80% of Generated matched selected = 8,150
val = stratified 20% of Generated matched selected = 2,039
test = Real matched selected = 10,189
```

Number of trained models:

```text
1
```

## 3. Q2 Real -> Generated-Balanced

Goal: train only on real data and test on balanced generated data. The training total stays aligned with Q1.

```text
Real matched selected = 10,189
Generated balanced selected = 10,189
```

Per group:

```text
Real: uses the matched real distribution, not balanced
Generated: 11 groups have 254, 29 groups have 255
```

Split:

```text
train = stratified 80% of Real matched selected = 8,150
val = stratified 20% of Real matched selected = 2,039
test = Generated balanced selected = 10,189
```

Number of trained models:

```text
1
```

## 4. Q2 Generated-Balanced -> Real

Goal: train only on balanced generated data and test on real data.

```text
Generated balanced selected = 10,189
Real matched selected = 10,189
```

Per group:

```text
Generated: 11 groups have 254, 29 groups have 255
Real: uses the matched real distribution, not balanced
```

Split:

```text
train = stratified 80% of Generated balanced selected = 8,149
val = stratified 20% of Generated balanced selected = 2,040
test = Real matched selected = 10,189
```

Number of trained models:

```text
1
```

## 5. Q3 Mixed Matched

Goal: train on a real/generated mixture while keeping the total training size unchanged. The generated distribution matches the real distribution.

Q3 trains one mixed model, not two models.

Starting from the Q1 matched data:

```text
Real matched pool = 10,189
Generated matched pool = 10,189
```

To make mixed train equal to separated train, real and generated data each contribute approximately half:

```text
mixed_train_total = 8,150
real_train_part = 4,075
generated_train_part = 4,075
```

Validation:

```text
mixed_val_total = 2,039
real_val_part = 1,019
generated_val_part = 1,020
```

Test:

```text
mixed_test_total = 10,189
real_test_part = 5,095
generated_test_part = 5,094
```

Plots:

```text
One Real Test curve
One Generated Test curve
```

Number of trained models:

```text
1
```

## 6. Q4 Mixed Peak-Valley Balanced

Goal: apply peak-valley balancing in mixed training while keeping train, validation, and test totals aligned with the other experiments.

Q4 trains one mixed model.

Full balanced target:

```text
mixed_total = 10,189
target per age_bin x gender group = 254 or 255
```

Train target:

```text
mixed_train_total = 8,150
8,150 / 40 = 203 remainder 30
10 groups have 203
30 groups have 204
```

Validation target:

```text
mixed_val_total = 2,039
2,039 / 40 = 50 remainder 39
1 group has 50
39 groups have 51
```

Test target:

```text
mixed_test_total = 10,189
10,189 / 40 = 254 remainder 29
11 groups have 254
29 groups have 255
```

Within each group:

```text
target real_part ~= group target / 2
target generated_part = group target - real_part
```

If real data exceeds `real_part`:

```text
Peak reduction: downsample real to real_part
Draw generated_part from generated data
```

If real data is below `real_part`:

```text
Valley filling: keep all real samples
Draw group target - real count from generated data
```

Current generated result:

```text
train = Real 2,090 + Generated 6,060 = 8,150
val = Real 521 + Generated 1,518 = 2,039
test = Real 2,754 + Generated 7,435 = 10,189
```

Plots:

```text
One Real Test curve
One Generated Test curve
```

Number of trained models:

```text
1
```

## Final Number of Models

| Quadrant | Experiment | Number of models |
| --- | --- | ---: |
| Q1 | Real -> Generated; Generated -> Real | 2 |
| Q2 | Real -> Generated-Balanced; Generated-Balanced -> Real | 2 |
| Q3 | Mixed Matched | 1 |
| Q4 | Mixed Peak-Valley Balanced | 1 |
| Total | 6 experiments | 6 |

## Final Scale Alignment

| Experiment type | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Separated | `8,149-8,150` | `2,039-2,040` | `10,189` |
| Mixed | `8,150` | `2,039` | `10,189` |

With this design, reviewers see that:

- All models have essentially the same training-set size.
- All models have essentially the same validation-set size.
- All models have the same test-set size.
- Differences come from data source and distribution strategy, not sample count.
