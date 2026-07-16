# PRD v2: Balanced Real/Generated Cross-Domain Experiments

This document is the English version of `PRD_balanced_real_generated_cross_domain.zh.md`. It records the second design iteration, where real and generated MRI are compared under controlled sample-count and distribution settings.

## Objective

The goal of v2 is to measure cross-domain generalization between real MRI and generated MRI while reducing confounding from sample count and age/sex imbalance.

The core questions are:

- How well does a model trained on real MRI generalize to generated MRI?
- How well does a model trained on generated MRI generalize to real MRI?
- How much of the observed gap is caused by the generated domain itself versus its age/sex distribution?

## Experimental Conditions

The v2 design introduces two separated-training directions:

1. `real -> generated`
   - Train on real MRI.
   - Validate on held-out real MRI.
   - Test on generated MRI.

2. `generated -> real`
   - Train on generated MRI.
   - Validate on held-out generated MRI.
   - Test on real MRI.

Generated data can be selected either to match the real distribution or to form a balanced distribution over `age_bin x gender` groups.

## Data Balancing

All balancing is performed over the joint `age_bin x gender` group, not over age or gender independently.

The matched condition uses:

```text
selected_count(group) = min(real_count(group), generated_count(group))
```

The generated-balanced condition keeps the generated total aligned with the real matched total, then distributes generated samples as evenly as possible across the 40 joint groups.

## Train / Validation / Test Rules

Separated experiments use a stratified train/validation split within the training domain:

```text
train ~= 80%
validation ~= 20%
```

The test domain is held out as the cross-domain target. The design avoids making one condition easier only because it has more training or test samples.

## Model and Metrics

The model is a PyTorch SFCN classifier over 5-year age bins. Training uses cross-entropy loss. Evaluation records:

- validation loss
- age-bin classification accuracy
- predicted age-bin center
- absolute error
- squared error
- sex-stratified and age-bin-stratified metrics

## Outputs

Each experiment writes:

- manifest CSV files under `manifests/`
- training logs and configuration under `logs/`
- best checkpoint under `checkpoints/best_model.pt`
- prediction CSV files under `predictions/`
- MAE/count figures under `figures/`

For public release, model checkpoints and logs are ignored by git; manifests, predictions, and figures remain available for reproducibility.

## Overwrite Policy

The workflow is designed to be restartable:

- Existing manifests are preserved unless `build-manifests --force` is used.
- Existing checkpoints skip training unless `train --force` is used.
- Existing prediction CSV files skip inference unless `infer --force` is used.
- Existing figures skip plotting unless `plot --force` is used.

## Evolution to v3

The v2 design clarified the separated-training cross-domain comparison, but it did not fully cover real/generated augmentation. The v3 design extends the experiment to four quadrants:

- separated + matched
- separated + generated-balanced
- mixed + matched
- mixed + peak-valley balanced
