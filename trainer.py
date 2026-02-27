import torch
import torch.nn as nn

from config import DEVICE


def _run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n


def train(model, train_loader, val_loader, epochs, lr, phase_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion)
        print(
            f"[{phase_name}] {epoch:>2}/{epochs}  "
            f"train  loss={train_loss:.4f}  acc={train_acc:.3f}  |  "
            f"val    loss={val_loss:.4f}  acc={val_acc:.3f}"
        )
