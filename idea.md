# Image Classification Idea: Wild Mushroom Species Classifier

## Problem Statement

Build an image classifier that identifies wild mushroom species and flags
whether they are **edible**, **toxic**, or **unknown/uncertain**.

Foragers, hikers, and survivalists often photograph mushrooms in the field
and need a quick safety assessment. A misidentification can be fatal — making
this a high-stakes, real-world problem worth solving.

## Why This Is Interesting

- **Safety-critical**: Amanita phalloides (Death Cap) kills dozens of people
  annually worldwide; a reliable classifier has genuine life-saving potential.
- **Visually challenging**: Many edible and toxic species look nearly identical
  (e.g., Amanita muscaria vs. Amanita caesarea). The model must learn fine-
  grained visual differences in cap color, gill structure, stem shape, and texture.
- **Long-tail distribution**: Rare species appear infrequently — a realistic
  class imbalance scenario.

## Transfer Learning Justification (ImageNet Link)

ImageNet contains classes such as `mushroom`, `agaric`, and various fungal
specimens. A pretrained CNN (e.g., ResNet-50, EfficientNet-B4) trained on
ImageNet already encodes:

- **Texture features** — crucial for distinguishing cap surfaces (smooth,
  scaly, slimy, fibrous)
- **Color gradients** — cap and gill coloring varies significantly by species
- **Edge/shape detectors** — useful for cap shape (convex, flat, umbonate)
  and gill attachment

Fine-tuning only the top layers on a mushroom dataset should converge much
faster than training from scratch, since early convolutional layers already
capture the relevant low-level features.

## Dataset

**Kaggle**: [Mushrooms Classification - Common Genus's Images](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)

- ~9 classes of common genera (Agaricus, Amanita, Boletus, Cortinarius, etc.)
- ~1,000–2,000 images per class
- RGB photos taken in natural forest settings

As a stretch, the [iNaturalist 2021](https://www.kaggle.com/c/inaturalist-2021)
dataset includes fungi as one of its supercategories and provides a much
larger, noisier, real-world challenge.

## Proposed Approach

1. **Baseline**: Fine-tune a pretrained ResNet-50 (ImageNet weights), freeze
   all layers except the final fully-connected head, train for ~10 epochs.
2. **Full fine-tune**: Unfreeze the last 2 residual blocks and continue
   training with a lower learning rate.
3. **Evaluation**: Accuracy, per-class F1, and a confusion matrix highlighting
   the most dangerous misclassifications (toxic predicted as edible).

## Success Criteria

- Top-1 accuracy ≥ 85% on a held-out test set
- Recall on toxic species ≥ 90% (false negatives are more dangerous than
  false positives in this domain)