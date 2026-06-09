# Deep Learning for Litter Detection: TACO Dataset

> Systematic comparison of object detection (YOLOv8) and semantic segmentation (UNet) for automated environmental monitoring using the TACO dataset.

---

## Results at a Glance

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| YOLOv8m — 640px | 7.90% | 4.20% | 29.38% | 12.97% |
| **YOLOv8m — 1280px** | **25.27%** | **15.72%** | **27.93%** | **31.09%** |
| YOLOv8s — 640px | 20.58% | 13.00% | 34.44% | 21.27% |
| UNet Exp. 1 (Baseline) | 17.78% mIoU | — | — | — |
| UNet Exp. 2 (Advanced) | 16.11% mIoU | — | — | — |
| UNet Exp. 3 (Conservative) | 9.43% mIoU | — | — | — |

**Key finding:** YOLOv8m at 1280px resolution achieves production-viable performance, with a 220% mAP improvement over the 640px baseline and cigarette detection jumping from 0.5% to 29.82% AP — solely through resolution scaling.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Experiments](#experiments)
  - [Object Detection — YOLOv8](#object-detection--yolov8)
  - [Semantic Segmentation — UNet](#semantic-segmentation--unet)
- [Per-Class Results](#per-class-results)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

Automated litter detection in urban and natural environments has significant potential for scalable waste monitoring. This project evaluates two deep learning paradigms — object detection and semantic segmentation — on the TACO (Trash Annotations in Context) dataset, with a particular focus on:

- Small object detection (cigarette butts: 2–10 pixels)
- The impact of input resolution on detection performance
- Handling extreme class imbalance (~95% background pixels)

---

## Dataset

**TACO (Trash Annotations in Context)**
- 1,500 high-resolution images
- 4,784 annotations across 59 categories (28 supercategories)
- Focus: top 5 most prevalent classes

| Class | Annotations | % of Dataset | Key Challenge |
|---|---|---|---|
| Cigarette | 667 | 13.94% | Extremely small (2–10px), similar to background |
| Unlabeled litter | 517 | 10.81% | Heterogeneous appearance, variable size |
| Plastic film | 451 | 9.43% | Semi-transparent, deformable shapes |
| Clear plastic bottle | 285 | 5.96% | Large but transparent, distinct edges |
| Other plastic | 273 | 5.71% | Diverse subcategories, mixed scales |

**Split strategy:** 80% train / 10% validation / 10% test (stratified random split)

**Core challenges:**
- Extreme class imbalance — background dominates ~95% of pixels
- Scale variation spanning two orders of magnitude (2px cigarettes → 200px+ bottles)
- Environmental variability across lighting conditions and contexts
- Annotation quality variations in segmentation ground truth

---

## Experiments

### Object Detection — YOLOv8

Three configurations were compared to isolate the effects of model size, resolution, and training strategy.

**Experiment 1 — YOLOv8m Baseline**
- Model: YOLOv8m (medium variant)
- Input resolution: 640×640
- Training: 40 epochs, batch 16, AdamW optimizer
- Augmentation: standard YOLO suite

**Experiment 2 — YOLOv8m High Resolution** ⭐ Best overall
- Model: YOLOv8m (same architecture)
- Input resolution: **1280×1280** (critical change)
- Training: 60 epochs, batch 4, conservative LR (0.001)
- Augmentation: optimised conservative strategy for small objects

**Experiment 3 — YOLOv8s Lightweight**
- Model: YOLOv8s (small/fast variant)
- Input resolution: 640×640
- Training: 75 epochs, SGD optimizer, multi-scale training

---

### Semantic Segmentation — UNet

Three UNet configurations with progressive optimisation, all using attention gates.

**Architecture:**
```
UNet: 3 → 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 6
```
All experiments used mixed precision (AMP), Adam (LR=1e-3), and early stopping.

**Experiment 1 — Baseline**
- Input: 512×512
- Loss: CE + Dice + Focal (0.4 : 0.4 : 0.2)
- Training: 20 epochs, batch 8, moderate class weights

**Experiment 2 — Advanced Loss & Augmentation**
- Loss: CE + Focal + Tversky + Dice (0.2 : 0.3 : 0.3 : 0.2)
- Tversky: α=0.7, β=0.3 (heavy false-negative penalty)
- Focal: γ=3 (focus on hard examples)
- Class weights: Background=0.02, Cigarette=2× boost
- Augmentation: RandomScale with 1.5× zoom-in, random cropping

**Experiment 3 — Conservative Small Object Preservation**
- No random cropping — LongestMaxSize + PadIfNeeded
- Minimal geometric transforms: rotation ≤30°, gentle shift/scale
- Balanced multi-component loss

---

## Per-Class Results

### Detection AP@0.5 by Resolution

| Class | 640px | 1280px | Improvement |
|---|---|---|---|
| Cigarette | 0.50% | **29.82%** | +5864% |
| Unlabeled litter | 7.27% | 14.81% | +104% |
| Plastic film | 4.16% | 16.02% | +285% |
| Clear plastic bottle | 20.42% | **59.29%** | +190% |
| Other plastic | 7.15% | 6.44% | −10% |

### Segmentation IoU — UNet Experiments

| Class | Exp. 1 | Exp. 2 | Exp. 3 |
|---|---|---|---|
| Background | 97.66% | 83.47% | 50.78% |
| Cigarette | 0.00% | 0.00% | 0.06% |
| Unlabeled litter | 0.00% | 0.12% | 0.31% |
| Plastic film | 3.01% | 5.84% | 4.22% |
| Clear plastic bottle | 5.99% | 6.95% | 1.12% |
| Other plastic | 0.01% | 0.27% | 0.09% |
| **Mean (litter only)** | **1.80%** | **2.64%** | **1.16%** |

---

## Key Findings

**1. Resolution is the dominant factor for small object detection**

The jump from 640px → 1280px produced a 220% improvement in overall mAP@0.5 and a 5864% improvement in cigarette AP. There appears to be a critical resolution threshold (~1280px) below which detecting 2–10 pixel objects is practically impossible.

**2. Object detection is fundamentally more suitable than segmentation for this task**

Object detection outperformed segmentation by 42% in headline metrics, but more critically, detection achieved *viable* performance on the hardest class while segmentation achieved essentially zero. The reasons:

- Bounding box predictions are more robust to tiny objects than pixel masks
- YOLO's multi-scale feature processing inherently handles scale variation
- UNet's encoder-decoder bottleneck loses fine spatial detail through downsampling
- Exact boundary delineation is impossible for 2–10 pixel objects

**3. Loss function engineering cannot overcome architectural limitations**

Despite state-of-the-art loss combinations (Tversky α=0.7, Focal γ=3, aggressive class weighting), UNet training plateaued at epochs 8–10 with no recovery. The problem is architectural capacity, not loss design.

**4. Pixel accuracy is a misleading metric under extreme class imbalance**

UNet Experiment 1 achieved 97.49% pixel accuracy while completely failing on all litter classes (cigarette IoU = 0.00%). High pixel accuracy reflects the model learning to predict "background" everywhere — not meaningful detection.

**5. Augmentation strategy matters for small objects**

Conservative augmentation (reduced scale variation, minimal rotation, no random cropping) was necessary to preserve the minimal pixel information available in cigarette-scale objects. Reductions applied:
- Scale augmentation: 0.5 → 0.2 (75% reduction)
- Rotation: 10° → 5°
- Translation: 0.1 → 0.05
- Mosaic probability: 1.0 → 0.3

---

## Limitations

- Dataset limited to 1,500 images — may not capture full environmental diversity
- Resolutions above 1280px not evaluated due to GPU memory constraints
- Inference speed / real-time deployment analysis not conducted
- Advanced segmentation architectures (DeepLab, SegFormer) not benchmarked

**Future directions:**
- Ultra-high resolution processing (2048px+) with model parallelisation
- Specialised small object detectors (FCOS, CenterNet)
- Multi-stage detection pipelines combining coarse and fine-grained models
- Edge computing deployment studies

---

## References

1. Proença, P.F., Simões, P. — *TACO: Trash Annotations in Context Dataset.* arXiv:2003.06975 (2020)
2. Ultralytics — *YOLOv8: A New State-of-the-Art Computer Vision Model.* github.com/ultralytics/ultralytics (2023)
3. Ronneberger, O., Fischer, P., Brox, T. — *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015
4. Chen, C. et al. — *R-CNN for Small Object Detection.* ACCV 2016
5. Lin, T.Y. et al. — *Focal Loss for Dense Object Detection.* ICCV 2017
6. Oktay, O. et al. — *Attention U-Net: Learning Where to Look for the Pancreas.* arXiv:1804.03999 (2018)
7. Salehi, S.S.M. et al. — *Tversky Loss Function for Image Segmentation.* MLMI 2017
8. Zhang, W. et al. — *Deep Learning for Environmental Monitoring: A Comprehensive Survey.* IEEE TNNLS 33(11) (2022)
