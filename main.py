import torch

from config import (DATA_DIR, DEVICE, LR_FINETUNE, LR_FROZEN,
                    NUM_EPOCHS_FINETUNE, NUM_EPOCHS_FROZEN)
from data import build_dataloaders, build_transforms, load_datasets
from download import download_dataset_if_needed
from evaluation import evaluate
from model import build_model, unfreeze_last_blocks
from stats import print_dataset_stats
from trainer import train


def main():
    download_dataset_if_needed(DATA_DIR)
    print_dataset_stats(DATA_DIR)

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
