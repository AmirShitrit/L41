# Mushroom Species Classifier

An image classifier that identifies wild mushroom species using transfer learning
on a ResNet-50 backbone pretrained on ImageNet.

## Problem

Wild mushrooms are notoriously difficult to identify. Many edible and toxic species
look nearly identical — a misidentification can be fatal. This classifier maps a
mushroom photo to one of 9 genera and flags whether it is typically edible or toxic.

**Classes:** Agaricus, Amanita, Boletus, Cortinarius, Entoloma, Hygrocybe,
Lactarius, Russula, Suillus

## Why Transfer Learning?

ImageNet-pretrained ResNet-50 already encodes texture gradients, color patterns,
and edge detectors that are directly relevant to distinguishing mushroom species
(cap surface, gill structure, stem shape). Fine-tuning converges far faster than
training from scratch on a ~6,700-image dataset.

## Dataset

[Mushrooms Classification - Common Genus's Images](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)
on Kaggle — ~6,700 RGB photos across 9 genera, taken in natural settings.
The dataset is **imbalanced** (ratio ≈ 0.20); minority classes have ~300 samples,
majority classes have ~1,500.

The dataset is downloaded automatically on first run (requires a Kaggle API token).

## Setup

```bash
# Install dependencies
uv pip install -r requirements.txt

# Add your Kaggle API token to .env
echo "KAGGLE_API_TOKEN=<your_token>" > .env
# Token available at: https://www.kaggle.com/settings → API → Create New Token

# Run
uv run main.py
```

## Training Pipeline

Training proceeds in two phases:

| Phase | Layers trained | Epochs | Learning rate |
|-------|---------------|--------|---------------|
| 1 — Head only | `fc` only (backbone frozen) | 10 | 1e-3 |
| 2 — Fine-tune | `layer3` + `layer4` + `fc` | 10 | 1e-4 |

## Outputs

| File | Description |
|------|-------------|
| `mushroom_classifier.pth` | Trained model weights |
| `confusion_matrix.png` | Per-class confusion matrix on the validation set |

## Project Structure

```
├── main.py          # Orchestration
├── config.py        # Hyperparameters and device
├── data.py          # Transforms, dataset loading, dataloaders
├── model.py         # ResNet-50 construction and layer unfreezing
├── trainer.py       # Training loop
├── evaluation.py    # Metrics and confusion matrix
├── stats.py         # Dataset statistics report
├── download.py      # Kaggle dataset download
└── requirements.txt
```
