# PRD v1: Real-Train / Generated-Validation SFCN Experiment

This document is the English version of `PRD_real_train_generated_val-v1.zh.md`. It records the first design iteration for comparing real MRI and generated MRI with an SFCN age-bin classifier.

## Objective

Build a reproducible PyTorch SFCN workflow that trains on real MRI and evaluates on generated MRI. The first version focuses on a direct cross-domain comparison:

- Train domain: real MRI.
- Validation/test domain: generated MRI.
- Task: age-bin classification.
- Output analysis: age-bin prediction error and sex-stratified metrics.

## Data Inputs

Real MRI is read from the configured real-data root and matched with metadata from the mapping table. Generated MRI is read from the generated-data root, with age and sex parsed from filenames such as:

```text
age1.00_sexM_s131.nii.gz
0004_age0.00_sexF_s4.nii.gz
```

Generated NIfTI headers are not treated as reliable metadata. The array is used as image data, while age and sex are derived from filenames.

## Age-Bin Task

The model predicts 5-year age bins from 5 to 105 years:

```text
5-10, 10-15, ..., 100-105
```

The classifier uses cross-entropy loss over age-bin classes. For analysis, predicted classes are mapped back to age-bin centers so MAE/MSE can be computed against the original age value.

## Workflow

The v1 workflow contains four major stages:

1. Build real and generated manifests.
2. Train the SFCN classifier on real MRI.
3. Run inference on generated MRI.
4. Aggregate prediction errors and plot age-bin metrics.

The design later evolved into the four-quadrant setup now implemented by `config.yaml`, `main.py`, and the modules under `src/sfcn/`.

## Training Defaults

The current release configuration supersedes early v1 experimental defaults:

- `batch_size = 8`
- `learning_rate = 1e-4`
- `patience = 5`
- `max_epochs = 1000`
- `weight_decay = 1e-4`
- `device = cuda`, with automatic fallback to `mps` and then `cpu`

## Reproducibility Requirements

- All sample selection should be derived from manifest CSV files.
- Random sampling must use a fixed seed.
- Training should save the exact configuration used for the run.
- Existing checkpoints, predictions, and figures should not be overwritten unless `--force` is explicitly provided.

## Limitations of v1

The v1 design is intentionally simple and does not yet fully separate distribution effects from domain effects. In particular:

- It does not enforce matched age/sex distributions between real and generated samples.
- It does not include a generated-balanced condition.
- It does not test mixed real/generated augmentation.

These limitations motivated the later v2 and v3 four-quadrant designs.
