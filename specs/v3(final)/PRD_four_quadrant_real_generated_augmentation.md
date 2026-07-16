# PRD v3: Four-Quadrant Real/Generated Augmentation Experiments

This document is the English version of `PRD_four_quadrant_real_generated_augmentation.zh.md`. It describes the final four-quadrant experimental design implemented in this repository.

## Objective

The goal is to compare real MRI and generated MRI in age-bin classification while separating the effects of:

- data domain: real versus generated
- training setup: separated versus mixed
- distribution strategy: matched, generated-balanced, or peak-valley balanced
- sample count: controlled across experiments

The final setup trains 6 models across Q1-Q4.

## Quadrants

| Quadrant | Training setup | Distribution setup | Purpose |
| --- | --- | --- | --- |
| Q1 | Separated training | Matched age/sex distribution | Controlled real/generated cross-domain baseline |
| Q2 | Separated training | Generated-balanced distribution | Tests generated data when age/sex groups are balanced |
| Q3 | Mixed training | Matched age/sex distribution | Ideal augmentation setting with matched real/generated distributions |
| Q4 | Mixed training | Peak-valley balanced distribution | Production-like augmentation setting using generated data to fill real-data gaps |

## Final Experiments

| No. | Experiment | Model |
| --- | --- | --- |
| 1 | Q1 Real -> Generated | Train real, test generated |
| 2 | Q1 Generated -> Real | Train generated, test real |
| 3 | Q2 Real -> Generated-Balanced | Train real, test balanced generated |
| 4 | Q2 Generated-Balanced -> Real | Train balanced generated, test real |
| 5 | Q3 Mixed Matched | Train one mixed real/generated model |
| 6 | Q4 Mixed Peak-Valley Balanced | Train one mixed real/generated model |

## Count Alignment

The design keeps train, validation, and test sizes aligned so that differences are attributable to data source and distribution strategy rather than sample count.

Current manifest totals:

| Experiment type | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Separated | `8,149-8,150` | `2,039-2,040` | `10,189` |
| Mixed | `8,150` | `2,039` | `10,189` |

## Distribution Rules

Matched distribution:

```text
generated_count(age_bin, gender) = real_count(age_bin, gender)
```

Generated-balanced distribution:

```text
10,189 samples over 40 age_bin x gender groups
11 groups have 254 samples
29 groups have 255 samples
```

Peak-valley mixed distribution:

```text
Downsample real-data peaks.
Keep real-data valleys.
Use generated data to fill the remaining target count.
```

All sampling is deterministic under `seed = 42`.

## Manifest Outputs

The implemented workflow stores experiment-specific manifests under `outputs/<quadrant>/<experiment>/manifests/` or `outputs/<quadrant>/manifests/`.

Examples:

```text
outputs/q1_separated_matched/real-gen/manifests/real_train_manifest.csv
outputs/q1_separated_matched/real-gen/manifests/real_val_manifest.csv
outputs/q1_separated_matched/real-gen/manifests/generated_test_manifest.csv
outputs/q3_mixed_matched/manifests/mixed_train_manifest.csv
outputs/q4_mixed_peak_valley/manifests/mixed_train_manifest.csv
```

## Training and Evaluation

The shared command entry point is:

```bash
python3 main.py <command> ...
```

Core commands:

```bash
python3 main.py build-manifests --quadrant all --force
python3 main.py train --experiment <config-name>
python3 main.py infer --experiment <config-name> --split <split-name>
python3 main.py plot --experiment <config-name>
python3 main.py scan-status --only-missing
```

Each command reads experiment configuration from `config.yaml`.

## Plotting Requirements

- Real data is green.
- Generated data is orange-red.
- MAE is plotted as a line.
- Sample count is plotted as bars.
- Separated experiments show training-domain counts as `Train + Validation`.
- Mixed experiments show mixed training counts as `Real Train + Generated Train`.
- Legends are placed outside the plot area.

## Release Policy

For public GitHub release:

- source code, manifests, predictions, and figures are kept
- checkpoints (`*.pt`) are ignored
- logs are ignored
- Python caches and OS metadata files are ignored

This keeps the repository reproducible without publishing model weights or local runtime artifacts.
