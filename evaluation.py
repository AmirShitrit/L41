import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import DEVICE


def _collect_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(DEVICE)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_labels, all_preds


def _save_confusion_matrix(labels, preds, class_names, path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Mushroom Classifier")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved to {path}")


def evaluate(model, loader, class_names):
    labels, preds = _collect_predictions(model, loader)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    _save_confusion_matrix(labels, preds, class_names)
