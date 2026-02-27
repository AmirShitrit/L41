import os

import torch

from config import (DATA_DIR, DEVICE, LR_FINETUNE, LR_FROZEN,
                    NUM_EPOCHS_FINETUNE, NUM_EPOCHS_FROZEN)
from data import build_dataloaders, build_transforms, load_datasets
from evaluation import evaluate
from model import build_model, unfreeze_last_blocks
from trainer import train


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"Dataset not found at '{DATA_DIR}'.")
        print("Download from: https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images")
        print(f"Place it so that '{DATA_DIR}/<class_name>/*.jpg' structure is satisfied.")
        return

    train_tf, val_tf = build_transforms()
    train_set, val_set, class_names = load_datasets(DATA_DIR, train_tf, val_tf)
    train_loader, val_loader = build_dataloaders(train_set, val_set)

    model = build_model(len(class_names))

    print(f"Classes  : {class_names}")
    print(f"Train/Val: {len(train_set)} / {len(val_set)} images")
    print(f"Device   : {DEVICE}\n")

    print("=== Phase 1: Head-only training (backbone frozen) ===")
    train(model, train_loader, val_loader, NUM_EPOCHS_FROZEN, LR_FROZEN, "frozen ")

    print("\n=== Phase 2: Fine-tuning (layer3 + layer4 + fc unfrozen) ===")
    unfreeze_last_blocks(model)
    train(model, train_loader, val_loader, NUM_EPOCHS_FINETUNE, LR_FINETUNE, "finetune")

    print("\n=== Final Evaluation ===")
    evaluate(model, val_loader, class_names)

    torch.save(model.state_dict(), "mushroom_classifier.pth")
    print("Model weights saved to mushroom_classifier.pth")


if __name__ == "__main__":
    main()
