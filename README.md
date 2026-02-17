# Soccer Player Detection: Classification + Conditional Localization

This repo implements an end-to-end **vision pipeline for a soccer-playing robot** using the *Soccer Player Detection* dataset. The system learns to understand a cluttered game scene from raw RGB images by solving two complementary tasks:

1. **Classification:** identify which entity is present (e.g., robot / ball / goalpost).
2. **Regression (Localization):** predict the **normalized center coordinates (x, y)** of a requested target object in the full camera frame. :contentReference[oaicite:0]{index=0}

---

## What the Project Does

### Task 1 — Object Classification (Cropped Inputs)
A lightweight CNN classifies cropped image regions into object categories. The pipeline uses **stratified splitting** to preserve minority classes and evaluates with accuracy. :contentReference[oaicite:1]{index=1}

### Task 2 — Object Localization (Full-Image Regression)
A regression model predicts the target’s center point **(cx, cy)** in **[0, 1] × [0, 1]**, derived from bounding boxes. To resolve ambiguity in scenes with multiple objects, the localization network is **conditional**: it takes both the image and a **target class ID** as inputs, so it can “focus” on the requested object. :contentReference[oaicite:2]{index=2}

---

## Key Points

### Preventing Data Leakage (Grouped Split)
For localization, splitting is done **at the image level** (grouped by filename), ensuring objects from the same original frame never appear across train/val/test—preventing background memorization and leakage. :contentReference[oaicite:3]{index=3}

### Regression-Safe Online Augmentation
A custom augmentation pipeline applies:
- photometric noise (brightness/contrast)
- horizontal flips **with synchronized label updates** (flipping `cx` → `1 - cx`)  
This improves stability and reduces overfitting in regression training. 

---

## Models

### Classification CNN
- Input: 128×128 RGB
- 3 Conv blocks (increasing filters), max-pooling
- Dense head + Softmax output :contentReference[oaicite:5]{index=5}

### Conditional Localization Network
- Visual branch: CNN feature extractor
- Context branch: class ID → **Embedding**
- Fusion: concatenate features + embedding
- Output: 2D (x, y) with **Sigmoid** to keep predictions within image bounds :contentReference[oaicite:6]{index=6}

Optimization uses Adam with a cosine-decay learning-rate schedule and early stopping. :contentReference[oaicite:7]{index=7}

---

## Results Summary (from the report)

- **Classification:** ~**98.5%** test accuracy with the final optimized CNN. :contentReference[oaicite:8]{index=8}  
- **Localization:** average error reduced from ~**28 px** (baseline) to ~**21.8 px** with the conditional architecture (ResNet-style + regularization). :contentReference[oaicite:9]{index=9}  
- Per-class localization is hardest on the **Robot** due to shape/orientation variation. :contentReference[oaicite:10]{index=10}

---

## Repo Contents (typical)
- notebooks / scripts for training classification and regression models
- dataset loader with:
  - filename reconciliation
  - stratified split for classification
  - grouped split for regression
- augmentation pipeline that keeps labels consistent with transforms

---

## Notes / Limitations
While classification is near-perfect, localization error can still be significant for fine control, and heavy occlusion cases may remain challenging. Future improvements suggested include moving to a single-shot detector (e.g., YOLO-style) and temporal smoothing (e.g., Kalman filtering). :contentReference[oaicite:11]{index=11}