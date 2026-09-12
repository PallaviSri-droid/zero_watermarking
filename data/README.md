# Real Medical Data

This directory intentionally contains no patient data. Store licensed datasets outside Git and point the benchmark to a CSV manifest.

## Recommended journal evaluation design

Use at least one radiography dataset for the primary study and, when feasible, a second modality such as MRI or OCT for cross-modality generalization. Recent medical zero-watermarking studies have used chest X-ray, MRI, and OCT examples, so a multi-modality extension is useful for testing generalization rather than tuning to one modality.

## Manifest format

Create `data/manifests/<dataset>.csv` with:

```csv
image_id,path,group_id,modality,label
img_000001,relative/or/absolute/path.png,patient_0001,CXR,normal
img_000002,relative/or/absolute/path.png,patient_0002,CXR,pneumonia
```

`group_id` must be a patient or study identifier whenever the source dataset provides one. Do not split individual slices from the same patient across train/validation/test.

## Recommended benchmark tracks

1. **Development track:** a small, reproducible subset for debugging.
2. **Primary track:** the complete approved dataset under one fixed protocol.
3. **Cross-modality track:** an independently sourced modality to test generalization.
4. **Stress track:** stronger attack strengths and compound attacks, reported separately from the primary benchmark.

Never copy patient images into Git unless the dataset license explicitly permits redistribution. Record dataset name, version, access date, preprocessing, split seed, and manifest checksum in the experiment configuration.